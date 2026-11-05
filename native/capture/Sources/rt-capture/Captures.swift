import AVFoundation
import CaptureCore
import CoreGraphics
import ScreenCaptureKit

struct CapturePermissionError: Error {
    let message: String
    let exitCode: Int32

    static let screenRecording = CapturePermissionError(
        message:
            "Screen Recording permission is missing. Allow your terminal app in System "
            + "Settings > Privacy & Security > Screen & System Audio Recording, then restart it.",
        exitCode: 2)
    static let microphone = CapturePermissionError(
        message:
            "Microphone permission is missing. Allow your terminal app in System Settings > "
            + "Privacy & Security > Microphone, then restart it.",
        exitCode: 3)
}

/// Turns one device's buffers into records on stdout, and keeps the stream's clock moving
/// when the device stops delivering.
final class StreamOutput {
    private let name: String
    private let writer: RecordWriter
    private let resampler = Resampler()
    private let assembler: FrameAssembler
    // Device callbacks and the stall timer run on different threads, and a restarted
    // device may call from a new one.
    private let lock = NSLock()
    private var stalled = false
    private let stallTimer = DispatchSource.makeTimerSource(queue: .global(qos: .userInitiated))

    init(stream: CaptureStream, name: String, writer: RecordWriter, sessionStart: Double) {
        self.name = name
        self.writer = writer
        assembler = FrameAssembler(stream: stream, sessionStart: sessionStart)
        stallTimer.schedule(deadline: .now() + 0.25, repeating: 0.25)
        stallTimer.setEventHandler { [weak self] in self?.checkForStall() }
        stallTimer.resume()
    }

    func receive(_ buffer: AVAudioPCMBuffer, hostSeconds: Double) {
        lock.lock()
        defer { lock.unlock() }
        if stalled {
            stalled = false
            log("\(name) is delivering audio again")
        }
        do {
            let samples = try resampler.convert(buffer)
            writer.write(assembler.append(samples, hostSeconds: hostSeconds))
        } catch {
            fail("Audio conversion failed: \(error)")
        }
    }

    private func checkForStall() {
        lock.lock()
        defer { lock.unlock() }
        let silence = assembler.fillStall(hostSeconds: hostSecondsNow())
        if !silence.isEmpty && !stalled {
            stalled = true
            log("\(name) stopped delivering audio, so this part of the call won't be heard")
        }
        writer.write(silence)
    }
}

final class SystemAudioCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    private static let restartRetryNanoseconds: UInt64 = 2_000_000_000

    private let output: StreamOutput
    private let queue = DispatchQueue(label: "rt-capture.system")
    private var stream: SCStream?

    init(writer: RecordWriter, sessionStart: Double) {
        output = StreamOutput(
            stream: .system, name: "System audio", writer: writer, sessionStart: sessionStart)
    }

    func start() async throws {
        do {
            try await startStream()
        } catch is NoDisplayError {
            log("System audio can't start until a display is on, usually after unlocking the Mac")
            restart()
        }
    }

    /// The stream is tied to a display, so it can end when displays change or sleep. The
    /// call goes on regardless, so this keeps trying, and the stream's stall handling
    /// covers the silence meanwhile.
    private func restart() {
        Task {
            var lastMessage: String?
            while true {
                do {
                    try await startStream()
                    log("System audio capture started again")
                    return
                } catch let error as CapturePermissionError {
                    fail(error.message, code: error.exitCode)
                } catch {
                    if error.localizedDescription != lastMessage {
                        lastMessage = error.localizedDescription
                        log("Can't restart system audio yet, still trying: \(error.localizedDescription)")
                    }
                    try? await Task.sleep(nanoseconds: Self.restartRetryNanoseconds)
                }
            }
        }
    }

    private func startStream() async throws {
        guard CGPreflightScreenCaptureAccess() || CGRequestScreenCaptureAccess() else {
            throw CapturePermissionError.screenRecording
        }
        let content: SCShareableContent
        do {
            content = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: true)
        } catch let error as SCStreamError where error.code == .userDeclined {
            throw CapturePermissionError.screenRecording
        }
        guard let display = content.displays.first else {
            throw NoDisplayError()
        }

        let configuration = SCStreamConfiguration()
        configuration.capturesAudio = true
        configuration.excludesCurrentProcessAudio = true
        configuration.sampleRate = 48000
        configuration.channelCount = 2
        // The video can't be turned off, so make it as cheap as it gets.
        configuration.width = 2
        configuration.height = 2
        configuration.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        configuration.showsCursor = false

        let filter = SCContentFilter(display: display, excludingWindows: [])
        let stream = SCStream(filter: filter, configuration: configuration, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue)
        try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: queue)
        try await stream.startCapture()
        self.stream = stream
    }

    func stream(
        _ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of type: SCStreamOutputType
    ) {
        guard type == .audio, let buffer = Self.pcmBuffer(from: sampleBuffer) else { return }
        // Presentation times are on the host clock, the same one the microphone uses.
        output.receive(buffer, hostSeconds: sampleBuffer.presentationTimeStamp.seconds)
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        // The person chose to stop sharing, from the macOS menu bar.
        if (error as? SCStreamError)?.code == .userStopped {
            fail("System audio capture was stopped from the menu bar")
        }
        log("System audio capture stopped: \(error.localizedDescription). Restarting it")
        restart()
    }

    private static func pcmBuffer(from sampleBuffer: CMSampleBuffer) -> AVAudioPCMBuffer? {
        guard let description = sampleBuffer.formatDescription else { return nil }
        let format = AVAudioFormat(cmAudioFormatDescription: description)
        let frames = AVAudioFrameCount(CMSampleBufferGetNumSamples(sampleBuffer))
        guard frames > 0,
            let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)
        else { return nil }
        buffer.frameLength = frames
        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer, at: 0, frameCount: Int32(frames), into: buffer.mutableAudioBufferList)
        return status == noErr ? buffer : nil
    }
}

/// ScreenCaptureKit only offers audio through a display, and reports none while they sleep.
struct NoDisplayError: Error, LocalizedError {
    var errorDescription: String? { "no display is on" }
}

enum MicrophoneError: Error, LocalizedError {
    case noDevice

    var errorDescription: String? { "No microphone is available" }
}

final class MicrophoneCapture {
    private static let restartRetrySeconds = 1.0

    private var engine: AVAudioEngine?
    private let output: StreamOutput
    private var observer: NSObjectProtocol?

    init(writer: RecordWriter, sessionStart: Double) {
        output = StreamOutput(
            stream: .mic, name: "Microphone", writer: writer, sessionStart: sessionStart)
    }

    static func requestPermission() async throws {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return
        case .notDetermined:
            log("Waiting for you to answer the macOS Microphone permission dialog")
            if await AVCaptureDevice.requestAccess(for: .audio) { return }
            throw CapturePermissionError.microphone
        default:
            throw CapturePermissionError.microphone
        }
    }

    func start() throws {
        try startEngine()
    }

    /// Plugging in a headset stops the engine. A new engine is used for the new device,
    /// because the old one can keep a format the hardware no longer has, and installing a
    /// tap with that raises an exception nothing can catch. The device can also be
    /// missing for a moment during the switch, so this keeps trying, and the stream's
    /// stall handling covers the silence meanwhile.
    private func restart() {
        if let engine {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        do {
            try startEngine()
            log("Microphone restarted")
        } catch {
            log("Can't restart the microphone yet: \(error.localizedDescription)")
            DispatchQueue.main.asyncAfter(deadline: .now() + Self.restartRetrySeconds) { [weak self] in
                self?.restart()
            }
        }
    }

    private func startEngine() throws {
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw MicrophoneError.noDevice
        }
        if let observer {
            NotificationCenter.default.removeObserver(observer)
        }
        // Registered before starting, so a change during the start isn't missed.
        observer = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main
        ) { [weak self] _ in
            log("Audio device changed, restarting the microphone")
            self?.restart()
        }
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [output] buffer, time in
            output.receive(buffer, hostSeconds: AVAudioTime.seconds(forHostTime: time.hostTime))
        }
        engine.prepare()
        try engine.start()
        self.engine = engine
    }
}
