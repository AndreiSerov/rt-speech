import XCTest
@testable import CaptureCore

final class FrameAssemblerTests: XCTestCase {
    private let frameSeconds = 512.0 / 16000.0

    private func assembler(sessionStart: Double = 100) -> FrameAssembler {
        FrameAssembler(stream: .mic, sessionStart: sessionStart)
    }

    private func ramp(_ count: Int, from start: Int = 0) -> [Float] {
        (0..<count).map { Float($0 + start) }
    }

    func testNothingIsEmittedBeforeAWholeFrame() {
        XCTAssertTrue(assembler().append(ramp(511), hostSeconds: 100).isEmpty)
    }

    func testSamplesAreCutIntoExactFramesAcrossAppends() {
        let frames = assembler()
        let first = frames.append(ramp(700), hostSeconds: 100)
        let second = frames.append(ramp(400, from: 700), hostSeconds: 100 + 700.0 / 16000)

        XCTAssertEqual(first.count, 1)
        XCTAssertEqual(second.count, 1)
        XCTAssertEqual(first[0].samples, ramp(512))
        XCTAssertEqual(second[0].samples, ramp(512, from: 512))
    }

    func testTimestampsCountFromTheSessionStart() {
        let frames = assembler(sessionStart: 100).append(ramp(1024), hostSeconds: 102.5)

        XCTAssertEqual(frames[0].timestamp, 2.5, accuracy: 1e-9)
        XCTAssertEqual(frames[1].timestamp, 2.5 + frameSeconds, accuracy: 1e-9)
    }

    func testAudioFromBeforeTheSessionStartClampsToZero() {
        let frames = assembler(sessionStart: 100).append(ramp(512), hostSeconds: 99)

        XCTAssertEqual(frames[0].timestamp, 0, accuracy: 1e-9)
    }

    func testSmallTimingJitterChangesNothing() {
        let frames = assembler()
        _ = frames.append(ramp(512), hostSeconds: 100)
        let next = frames.append(ramp(512, from: 512), hostSeconds: 100 + frameSeconds + 0.05)

        XCTAssertEqual(next[0].timestamp, frameSeconds, accuracy: 1e-9)
        XCTAssertEqual(next[0].samples, ramp(512, from: 512))
    }

    func testAGapIsFilledWithSilenceSoLaterTimestampsStayTrue() {
        let frames = assembler()
        _ = frames.append(ramp(512), hostSeconds: 100)
        // The next audio should arrive at 100.032, but arrives a whole second later.
        let later = frames.append(ramp(512), hostSeconds: 100 + frameSeconds + 1)

        XCTAssertEqual(later.count, 32)
        XCTAssertEqual(later[0].samples, [Float](repeating: 0, count: 512))
        let last = later[later.count - 1]
        XCTAssertEqual(last.timestamp, frameSeconds + 1, accuracy: 0.033)
        // 16000 filled samples end 128 into this frame, and the ramp starts at 0.
        XCTAssertEqual(last.samples.filter { $0 != 0 }.count, 383)
    }

    func testAVeryLongGapJumpsTheClockInsteadOfFillingIt() {
        let frames = assembler()
        _ = frames.append(ramp(512), hostSeconds: 100)
        // A Mac asleep for an hour mustn't produce an hour of silence.
        let later = frames.append(ramp(512), hostSeconds: 3700)

        XCTAssertEqual(later.count, 1)
        XCTAssertEqual(later[0].timestamp, 3600, accuracy: 1e-9)
    }

    func testAudioAheadOfTheClockIsDroppedSoDriftDoesNotAccumulate() {
        let frames = assembler()
        _ = frames.append(ramp(16000), hostSeconds: 100)
        // A whole second of audio arrived, but the clock says only 0.5 s has passed,
        // so 8000 of these new samples are ahead of real time.
        let next = frames.append(ramp(8512), hostSeconds: 100.5)

        let timestamps = next.map(\.timestamp)
        XCTAssertEqual(timestamps.first ?? -1, 0.992, accuracy: 0.001)
        XCTAssertEqual(next.count, 1)
    }

    func testRecordsHaveTheAgreedBytes() {
        let record = Record(stream: .mic, timestamp: 1.5, samples: [Float](repeating: 0.25, count: 512))

        let bytes = [UInt8](record.encoded())

        XCTAssertEqual(bytes.count, 2058)
        XCTAssertEqual(bytes[0], 1)  // Format version.
        XCTAssertEqual(bytes[1], 1)  // Mic.
        // 1.5 as little-endian float64.
        XCTAssertEqual(Array(bytes[2..<10]), [0, 0, 0, 0, 0, 0, 0xF8, 0x3F])
        // 0.25 as little-endian float32.
        XCTAssertEqual(Array(bytes[10..<14]), [0, 0, 0x80, 0x3E])
    }

    func testTheSystemStreamIsZero() {
        let record = Record(stream: .system, timestamp: 0, samples: [Float](repeating: 0, count: 512))

        XCTAssertEqual([UInt8](record.encoded())[1], 0)
    }
}
