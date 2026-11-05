import Foundation

public let sampleRate = 16000.0
public let frameSamples = 512
private let formatVersion: UInt8 = 1

/// The values match the wire format, so don't reorder them.
public enum CaptureStream: UInt8 {
    case system = 0
    case mic = 1
}

/// One frame of audio, as written to stdout.
public struct Record {
    public let stream: CaptureStream
    public let timestamp: Double
    public let samples: [Float]

    public init(stream: CaptureStream, timestamp: Double, samples: [Float]) {
        self.stream = stream
        self.timestamp = timestamp
        self.samples = samples
    }

    /// version u8, stream u8, timestamp f64, then samples f32, all little-endian.
    public func encoded() -> Data {
        var data = Data(capacity: 10 + samples.count * 4)
        data.append(formatVersion)
        data.append(stream.rawValue)
        withUnsafeBytes(of: timestamp.bitPattern.littleEndian) { data.append(contentsOf: $0) }
        for sample in samples {
            withUnsafeBytes(of: sample.bitPattern.littleEndian) { data.append(contentsOf: $0) }
        }
        return data
    }
}

/// Cuts one stream's audio into frames, timestamped on the clock shared by all streams.
///
/// Timestamps come from counting samples, which is smooth, but are kept honest against
/// the host clock that arrives with each buffer: gaps are filled with silence, because
/// the pipeline treats frames as contiguous, and audio ahead of the clock is dropped,
/// so device clock drift can't accumulate on a long call.
public final class FrameAssembler {
    /// Differences below this are normal delivery jitter.
    private static let toleratedDrift = 0.1
    /// Gaps above this are a sleeping Mac, not lost audio, so the clock simply jumps.
    private static let longestFilledGap = 5.0

    private let stream: CaptureStream
    private let sessionStart: Double
    private var origin: Double?
    private var emitted = 0
    private var pending: [Float] = []

    public init(stream: CaptureStream, sessionStart: Double) {
        self.stream = stream
        self.sessionStart = sessionStart
    }

    /// `hostSeconds` is when the first of `samples` was captured, on the host clock.
    public func append(_ samples: [Float], hostSeconds: Double) -> [Record] {
        let position = max(hostSeconds - sessionStart, 0)
        var incoming = samples[...]
        if let origin {
            let expected = origin + Double(emitted * frameSamples + pending.count) / sampleRate
            let gap = position - expected
            if gap > Self.longestFilledGap {
                self.origin = position
                emitted = 0
                pending.removeAll()
            } else if gap > Self.toleratedDrift {
                pending.append(contentsOf: repeatElement(0, count: Int((gap * sampleRate).rounded())))
            } else if gap < -Self.toleratedDrift {
                incoming = incoming.dropFirst(Int((-gap * sampleRate).rounded()))
            }
        } else {
            origin = position
        }
        pending.append(contentsOf: incoming)
        return takeWholeFrames()
    }

    private func takeWholeFrames() -> [Record] {
        guard let origin else { return [] }
        let count = pending.count / frameSamples
        var records: [Record] = []
        records.reserveCapacity(count)
        for index in 0..<count {
            let start = index * frameSamples
            let timestamp = origin + Double(emitted * frameSamples) / sampleRate
            records.append(
                Record(
                    stream: stream,
                    timestamp: timestamp,
                    samples: Array(pending[start..<(start + frameSamples)])))
            emitted += 1
        }
        pending.removeFirst(count * frameSamples)
        return records
    }
}
