import AVFoundation
import XCTest
@testable import CaptureCore

final class ResamplerTests: XCTestCase {
    private func tone(
        hertz: Double, rate: Double, channels: AVAudioChannelCount, seconds: Double
    ) -> AVAudioPCMBuffer {
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: channels, interleaved: false)!
        let count = AVAudioFrameCount(rate * seconds)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: count)!
        buffer.frameLength = count
        for channel in 0..<Int(channels) {
            for index in 0..<Int(count) {
                buffer.floatChannelData![channel][index] = Float(0.5 * sin(2 * .pi * hertz * Double(index) / rate))
            }
        }
        return buffer
    }

    /// The frequency, estimated from how often the signal crosses zero upward.
    private func hertz(of samples: [Float]) -> Double {
        var crossings = 0
        for index in 1..<samples.count where samples[index - 1] < 0 && samples[index] >= 0 {
            crossings += 1
        }
        return Double(crossings) / (Double(samples.count) / 16000)
    }

    func testStereo48kBecomes16kMonoAtTheSamePitch() throws {
        let resampler = Resampler()
        var samples: [Float] = []
        // A real stream arrives in small buffers, and the converter holds a little back.
        for _ in 0..<50 {
            samples += try resampler.convert(tone(hertz: 480, rate: 48000, channels: 2, seconds: 0.1))
        }

        XCTAssertEqual(Double(samples.count), 80000, accuracy: 500)
        let middle = Array(samples[20000..<40000])
        XCTAssertEqual(hertz(of: middle), 480, accuracy: 5)
        XCTAssertEqual(middle.map(abs).max() ?? 0, 0.5, accuracy: 0.1)
    }

    func testAudioHeldBackStaysSmallAtRealDeviceBufferSizes() throws {
        // Anything held back here is added to every Turn's latency.
        for (rate, size) in [(48000.0, 1024.0), (48000.0, 4800.0), (44100.0, 4410.0)] {
            let resampler = Resampler()
            var fedSeconds = 0.0
            var out = 0
            for _ in 0..<100 {
                out += try resampler.convert(
                    tone(hertz: 440, rate: rate, channels: 1, seconds: size / rate)
                ).count
                fedSeconds += size / rate
                if fedSeconds > 1 {
                    let heldSeconds = fedSeconds - Double(out) / 16000
                    XCTAssertLessThan(heldSeconds, 0.1, "\(rate) Hz, \(size) frames")
                }
            }
        }
    }

    func testMono44100BecomesAlmostExactlyTheRightLengthAcrossManyBuffers() throws {
        let resampler = Resampler()
        var total = 0
        for _ in 0..<100 {
            total += try resampler.convert(tone(hertz: 440, rate: 44100, channels: 1, seconds: 0.1)).count
        }

        XCTAssertEqual(Double(total), 160000, accuracy: 200)
    }

    func testAlready16kMonoKeepsItsLength() throws {
        let resampler = Resampler()
        var total = 0
        for _ in 0..<50 {
            total += try resampler.convert(tone(hertz: 440, rate: 16000, channels: 1, seconds: 0.1)).count
        }

        XCTAssertEqual(Double(total), 80000, accuracy: 400)
    }

    func testAudioInterfacesWithoutAChannelLayoutStayAudible() throws {
        // USB interfaces can report plain numbered channels, which can't be downmixed.
        for tag in [
            kAudioChannelLayoutTag_DiscreteInOrder | 2, kAudioChannelLayoutTag_DiscreteInOrder | 4,
            kAudioChannelLayoutTag_Unknown | 2, kAudioChannelLayoutTag_Unknown | 4,
        ] {
            let layout = AVAudioChannelLayout(layoutTag: tag)!
            XCTAssertGreaterThan(try peak(through: layout), 0.3, "layout tag \(tag)")
        }
    }

    func testChannelsDescribedOnlyByNumberStayAudible() throws {
        for label in [kAudioChannelLabel_Discrete_0, kAudioChannelLabel_Unused, kAudioChannelLabel_Unknown] {
            for channels in [2, 4] {
                XCTAssertGreaterThan(
                    try peak(through: describedLayout(labels: [AudioChannelLabel](repeating: label, count: channels))),
                    0.3, "\(channels) channels labeled \(label)")
            }
        }
    }

    func testChannelsDescribedAsSpeakersStillMixTogether() throws {
        let layout = describedLayout(labels: [kAudioChannelLabel_Left, kAudioChannelLabel_Right])
        XCTAssertFalse(Resampler.usesFirstChannelOnly(AVAudioFormat(standardFormatWithSampleRate: 48000, channelLayout: layout)))
    }

    private func describedLayout(labels: [AudioChannelLabel]) -> AVAudioChannelLayout {
        let size = MemoryLayout<AudioChannelLayout>.size + labels.count * MemoryLayout<AudioChannelDescription>.size
        let raw = UnsafeMutableRawPointer.allocate(byteCount: size, alignment: MemoryLayout<AudioChannelLayout>.alignment)
        defer { raw.deallocate() }
        raw.initializeMemory(as: UInt8.self, repeating: 0, count: size)
        let layout = raw.assumingMemoryBound(to: AudioChannelLayout.self)
        layout.pointee.mChannelLayoutTag = kAudioChannelLayoutTag_UseChannelDescriptions
        layout.pointee.mNumberChannelDescriptions = UInt32(labels.count)
        let descriptions = raw.advanced(by: MemoryLayout<AudioChannelLayout>.offset(of: \.mChannelDescriptions)!)
            .assumingMemoryBound(to: AudioChannelDescription.self)
        for (index, label) in labels.enumerated() {
            descriptions[index].mChannelLabel = label
        }
        return AVAudioChannelLayout(layout: layout)
    }

    /// The loudest output when only the first channel has a microphone connected.
    private func peak(through layout: AVAudioChannelLayout) throws -> Float {
        let format = AVAudioFormat(standardFormatWithSampleRate: 48000, channelLayout: layout)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4800)!
        buffer.frameLength = 4800
        for index in 0..<4800 {
            buffer.floatChannelData![0][index] = Float(0.5 * sin(2 * .pi * 480 * Double(index) / 48000))
        }
        let resampler = Resampler()
        var samples: [Float] = []
        for _ in 0..<5 { samples += try resampler.convert(buffer) }
        return samples.map(abs).max() ?? 0
    }

    func testAChangedInputFormatIsHandled() throws {
        // Plugging in a headset can change the mic's format mid-call.
        let resampler = Resampler()
        _ = try resampler.convert(tone(hertz: 440, rate: 48000, channels: 2, seconds: 0.5))

        var total = 0
        for _ in 0..<20 {
            total += try resampler.convert(tone(hertz: 440, rate: 24000, channels: 1, seconds: 0.1)).count
        }

        XCTAssertEqual(Double(total), 32000, accuracy: 400)
    }
}

final class ResamplerContinuityTests: XCTestCase {
    func testNoAudioIsLostAcrossBufferSizesTypicalOfRealDevices() throws {
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 48000, channels: 2, interleaved: false)!
        for size in [128, 480, 512, 1024, 4096, 32768] as [AVAudioFrameCount] {
            let resampler = Resampler()
            var total = 0
            var fed = 0
            while fed < 480000 {
                let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: size)!
                buffer.frameLength = size
                total += try resampler.convert(buffer).count
                fed += Int(size)
            }
            XCTAssertEqual(Double(total), Double(fed) / 3, accuracy: 200, "buffer size \(size)")
        }
    }
}
