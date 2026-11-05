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
/// the host clock that arrives with each buffer: gaps of any length are filled with
/// silence, because the pipeline treats frames as contiguous, and audio ahead of the
/// clock is dropped, so device clock drift can't accumulate on a long call. The host
/// clock stops while a Mac sleeps, so sleep doesn't look like a gap.
public final class FrameAssembler {
    /// Differences below this are normal delivery jitter.
    private static let toleratedDrift = 0.1
    /// A stream this far behind the clock has stopped delivering.
    private static let stallSeconds = 1.0
    /// A stall is filled only up to this far behind the clock, so audio that was merely
    /// late still fits when it arrives.
    private static let stallMargin = 0.5

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
            let gap = position - expected(from: origin)
            if gap > Self.toleratedDrift {
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

    /// Call regularly with the current host time. If the device has stopped delivering,
    /// returns silence so this stream's clock keeps moving: an open Segment or Turn can
    /// then close instead of waiting for the device to come back.
    public func fillStall(hostSeconds: Double) -> [Record] {
        guard let origin else { return [] }
        let position = hostSeconds - sessionStart
        let behind = position - expected(from: origin)
        guard behind > Self.stallSeconds else { return [] }
        let count = Int(((behind - Self.stallMargin) * sampleRate).rounded())
        pending.append(contentsOf: repeatElement(0, count: count))
        return takeWholeFrames()
    }

    /// Where the next sample belongs, in seconds since the session start.
    private func expected(from origin: Double) -> Double {
        origin + Double(emitted * frameSamples + pending.count) / sampleRate
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
