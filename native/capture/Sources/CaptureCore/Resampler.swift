import AVFoundation

public enum ResamplerError: Error {
    case unsupportedFormat(String)
    case conversionFailed(String)
}

/// Converts audio in any format to 16 kHz mono float32. It keeps its state between
/// calls, so consecutive buffers join without clicks.
public final class Resampler {
    private let target = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false)!
    private var converter: AVAudioConverter?

    public init() {}

    /// Channels that aren't speaker positions have no known meaning, and downmixing them
    /// silently gives all zeros. For those, the first channel is used.
    static func usesFirstChannelOnly(_ format: AVAudioFormat) -> Bool {
        guard format.channelCount > 1 else { return false }
        guard let layout = format.channelLayout else { return format.channelCount > 2 }
        if layout.layoutTag == kAudioChannelLayoutTag_UseChannelDescriptions {
            return !hasOnlySpeakerLabels(layout)
        }
        let kind = layout.layoutTag & 0xFFFF_0000
        return kind == kAudioChannelLayoutTag_DiscreteInOrder || kind == kAudioChannelLayoutTag_Unknown
    }

    private static func hasOnlySpeakerLabels(_ layout: AVAudioChannelLayout) -> Bool {
        let offset = MemoryLayout<AudioChannelLayout>.offset(of: \.mChannelDescriptions)!
        let first = UnsafeRawPointer(layout.layout).advanced(by: offset)
            .assumingMemoryBound(to: AudioChannelDescription.self)
        let descriptions = UnsafeBufferPointer(
            start: first, count: Int(layout.layout.pointee.mNumberChannelDescriptions))
        return descriptions.allSatisfy {
            (kAudioChannelLabel_Left..<kAudioChannelLabel_Discrete).contains($0.mChannelLabel)
        }
    }

    public func convert(_ input: AVAudioPCMBuffer) throws -> [Float] {
        if converter?.inputFormat != input.format {
            guard let new = AVAudioConverter(from: input.format, to: target) else {
                throw ResamplerError.unsupportedFormat("\(input.format)")
            }
            if Self.usesFirstChannelOnly(input.format) {
                new.channelMap = [0]
            } else {
                new.downmix = true
            }
            converter = new
        }
        guard let converter else { return [] }

        let ratio = target.sampleRate / input.format.sampleRate
        // Room for what this buffer yields, plus what the converter held back earlier.
        let capacity = AVAudioFrameCount(Double(input.frameLength) * ratio) + 1024
        var supplied = false
        var samples: [Float] = []
        samples.reserveCapacity(Int(capacity))

        // One call can return early with converted audio still held inside, even when it
        // reports running dry, so continue until a call produces nothing.
        while true {
            guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else {
                throw ResamplerError.conversionFailed("can't allocate \(capacity) frames")
            }
            var error: NSError?
            let status = converter.convert(to: output, error: &error) { _, inputStatus in
                if supplied {
                    inputStatus.pointee = .noDataNow
                    return nil
                }
                supplied = true
                inputStatus.pointee = .haveData
                return input
            }
            if status == .error {
                throw ResamplerError.conversionFailed(error?.localizedDescription ?? "unknown")
            }
            samples.append(
                contentsOf: UnsafeBufferPointer(
                    start: output.floatChannelData![0], count: Int(output.frameLength)))
            if output.frameLength == 0 {
                return samples
            }
        }
    }
}
