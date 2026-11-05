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

/// Turns one device's buffers into records on stdout.
final class StreamOutput {
    private let writer: RecordWriter
    private let resampler = Resampler()
    private let assembler: FrameAssembler

    init(stream: CaptureStream, writer: RecordWriter, sessionStart: Double) {
        self.writer = writer
        assembler = FrameAssembler(stream: stream, sessionStart: sessionStart)
    }

    func receive(_ buffer: AVAudioPCMBuffer, hostSeconds: Double) {
        do {
            let samples = try resampler.convert(buffer)
            writer.write(assembler.append(samples, hostSeconds: hostSeconds))
        } catch {
            fail("Audio conversion failed: \(error)")
        }
    }
}

final class SystemAudioCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    private let output: StreamOutput
    private let queue = DispatchQueue(label: "rt-capture.system")
    private var stream: SCStream?

    init(writer: RecordWriter, sessionStart: Double) {
        output = StreamOutput(stream: .system, writer: writer, sessionStart: sessionStart)
    }

    func start() async throws {
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
            throw CocoaError(.featureUnsupported)
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
        fail("System audio capture stopped: \(error.localizedDescription)")
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

final class MicrophoneCapture {
    private let engine = AVAudioEngine()
    private let output: StreamOutput
    private var observer: NSObjectProtocol?

    init(writer: RecordWriter, sessionStart: Double) {
        output = StreamOutput(stream: .mic, writer: writer, sessionStart: sessionStart)
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
        // Plugging in a headset stops the engine, so start it again on the new device.
        // The silence during the switch is filled in by the frame assembler.
        observer = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main
        ) { [weak self] _ in
            guard let self else { return }
            log("Audio device changed, restarting the microphone")
            self.engine.inputNode.removeTap(onBus: 0)
            do {
                try self.startEngine()
            } catch {
                fail("Microphone capture failed after a device change: \(error.localizedDescription)")
            }
        }
    }

    private func startEngine() throws {
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            fail("No microphone is available")
        }
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [output] buffer, time in
            output.receive(buffer, hostSeconds: AVAudioTime.seconds(forHostTime: time.hostTime))
        }
        engine.prepare()
        try engine.start()
    }
}
