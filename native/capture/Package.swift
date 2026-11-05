// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "rt-capture",
    platforms: [.macOS(.v13)],
    targets: [
        .target(name: "CaptureCore"),
        .executableTarget(
            name: "rt-capture",
            dependencies: ["CaptureCore"],
            exclude: ["Info.plist"],
            linkerSettings: [
                // A command-line tool has no app bundle, so its microphone usage
                // description has to be embedded in the binary.
                .unsafeFlags([
                    "-Xlinker", "-sectcreate", "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist", "-Xlinker", Context.packageDirectory + "/Sources/rt-capture/Info.plist",
                ])
            ]
        ),
        .testTarget(name: "CaptureCoreTests", dependencies: ["CaptureCore"]),
    ]
)
