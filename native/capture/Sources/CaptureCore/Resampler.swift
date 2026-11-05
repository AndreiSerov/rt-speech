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

    /// Plain numbered channels have no known meaning, and downmixing them silently gives
    /// all zeros. For those, the first channel is used.
    private static func canDownmix(_ format: AVAudioFormat) -> Bool {
        guard format.channelCount > 1 else { return true }
        guard let tag = format.channelLayout?.layoutTag else { return format.channelCount == 2 }
        let kind = tag & 0xFFFF_0000
        return kind != kAudioChannelLayoutTag_DiscreteInOrder && kind != kAudioChannelLayoutTag_Unknown
    }

    public func convert(_ input: AVAudioPCMBuffer) throws -> [Float] {
        if converter?.inputFormat != input.format {
            guard let new = AVAudioConverter(from: input.format, to: target) else {
                throw ResamplerError.unsupportedFormat("\(input.format)")
            }
            if Self.canDownmix(input.format) {
                new.downmix = true
            } else {
                new.channelMap = [0]
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
