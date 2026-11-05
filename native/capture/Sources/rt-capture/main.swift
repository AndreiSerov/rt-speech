// Captures system audio and the microphone as 16 kHz mono frames.
//
// Frames go to stdout as fixed-size records (see CaptureCore's Record), and logs go to
// stderr. Exit codes: 0 finished, 1 failed, 2 no Screen Recording permission, 3 no
// Microphone permission, 64 bad usage.
//
// --test-signal [--duration seconds]  Synthetic tones on both streams; needs no permission.
// --no-system, --no-mic               Capture only the other device, to diagnose permissions.

import AVFoundation
import CaptureCore
import Foundation

func log(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

func fail(_ message: String, code: Int32 = 1) -> Never {
    log(message)
    exit(code)
}

/// Serializes both streams' records onto stdout. A closed stdout ends the process with
/// the default SIGPIPE, so capture can't outlive whoever is reading it.
final class RecordWriter {
    private let queue = DispatchQueue(label: "rt-capture.writer")

    func write(_ records: [Record]) {
        guard !records.isEmpty else { return }
        queue.async {
            for record in records {
                record.encoded().withUnsafeBytes { bytes in
                    var offset = 0
                    while offset < bytes.count {
                        let written = Darwin.write(
                            STDOUT_FILENO, bytes.baseAddress! + offset, bytes.count - offset)
                        if written < 0 {
                            if errno == EINTR { continue }
                            exit(0)  // Nobody is reading any more.
                        }
                        offset += written
                    }
                }
            }
        }
    }

    /// Waits until everything written so far has reached stdout.
    func drain() {
        queue.sync {}
    }
}

func hostSecondsNow() -> Double {
    AVAudioTime.seconds(forHostTime: mach_absolute_time())
}

func runTestSignal(writer: RecordWriter, sessionStart: Double, duration: Double) -> Never {
    let frameSeconds = Double(frameSamples) / sampleRate
    let tones: [(CaptureStream, Double)] = [(.system, 440), (.mic, 880)]
    let assemblers = tones.map { FrameAssembler(stream: $0.0, sessionStart: sessionStart) }
    let frameCount = Int(duration / frameSeconds)
    for frame in 0..<frameCount {
        let host = sessionStart + Double(frame) * frameSeconds
        let wait = host - hostSecondsNow()
        if wait > 0 { usleep(UInt32(wait * 1_000_000)) }
        for (index, tone) in tones.enumerated() {
            let samples = (0..<frameSamples).map { sample -> Float in
                let time = Double(frame * frameSamples + sample) / sampleRate
                return Float(0.5 * sin(2 * .pi * tone.1 * time))
            }
            writer.write(assemblers[index].append(samples, hostSeconds: host))
        }
    }
    writer.drain()
    exit(0)
}

var system = true
var mic = true
var testSignal = false
var duration = 10.0

var arguments = CommandLine.arguments.dropFirst().makeIterator()
while let argument = arguments.next() {
    switch argument {
    case "--no-system": system = false
    case "--no-mic": mic = false
    case "--test-signal": testSignal = true
    case "--duration":
        guard let value = arguments.next().flatMap(Double.init), value > 0 else {
            fail("--duration needs a positive number of seconds", code: 64)
        }
        duration = value
    default:
        fail("unknown argument: \(argument)", code: 64)
    }
}
if !system && !mic {
    fail("--no-system and --no-mic leave nothing to capture", code: 64)
}

let writer = RecordWriter()
let sessionStart = hostSecondsNow()

if testSignal {
    runTestSignal(writer: writer, sessionStart: sessionStart, duration: duration)
}

// Kept alive for the whole run.
var systemCapture: SystemAudioCapture?
var micCapture: MicrophoneCapture?

Task {
    do {
        if mic {
            try await MicrophoneCapture.requestPermission()
            let capture = MicrophoneCapture(writer: writer, sessionStart: sessionStart)
            try capture.start()
            micCapture = capture
            log("Microphone capture started")
        }
        if system {
            let capture = SystemAudioCapture(writer: writer, sessionStart: sessionStart)
            try await capture.start()
            systemCapture = capture
        }
    } catch let error as CapturePermissionError {
        fail(error.message, code: error.exitCode)
    } catch {
        fail("Capture failed to start: \(error.localizedDescription)")
    }
}

dispatchMain()
