import AVFoundation
import CoreImage
import Foundation
import UIKit

@MainActor
final class StreamingViewModel: NSObject, ObservableObject {
    @Published var host: String = "192.168.0.10"
    @Published var port: String = "8080"
    @Published var flipH: Bool = false
    @Published var flipV: Bool = false
    @Published var isStreaming: Bool = false
    @Published var statusText: String = "Ready"
    @Published var senderFPS: Double = 0
    @Published var lastError: String?

    let captureSession = AVCaptureSession()

    private let output = AVCaptureVideoDataOutput()
    private let outputQueue = DispatchQueue(label: "PhoneStreamer.output", qos: .userInitiated)
    private let uploadStateQueue = DispatchQueue(label: "PhoneStreamer.uploadState")
    private let ciContext = CIContext(options: nil)
    private var isConfigured = false
    private var isUploading = false
    private var sentTimestamps: [TimeInterval] = []
    private var fpsTimer: Timer?
    private var appStateObserver: NSObjectProtocol?

    private lazy var urlSession: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 2.0
        config.timeoutIntervalForResource = 2.0
        return URLSession(configuration: config)
    }()

    var canStream: Bool {
        !host.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !port.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func onAppear() {
        guard appStateObserver == nil else { return }
        appStateObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.willResignActiveNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.stopStreaming(reason: "Paused (background)")
        }
        requestCameraPermissionAndConfigure()
    }

    func onDisappear() {
        stopStreaming(reason: "Stopped")
        captureSession.stopRunning()
        if let appStateObserver {
            NotificationCenter.default.removeObserver(appStateObserver)
        }
        appStateObserver = nil
    }

    func toggleStreaming() {
        isStreaming ? stopStreaming(reason: "Stopped") : startStreaming()
    }

    private func requestCameraPermissionAndConfigure() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureSessionIfNeeded()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                Task { @MainActor in
                    guard let self else { return }
                    if granted {
                        self.configureSessionIfNeeded()
                    } else {
                        self.statusText = "Camera permission denied"
                    }
                }
            }
        default:
            statusText = "Camera permission denied"
        }
    }

    private func configureSessionIfNeeded() {
        guard !isConfigured else {
            if !captureSession.isRunning {
                outputQueue.async { [weak self] in
                    self?.captureSession.startRunning()
                }
            }
            return
        }

        captureSession.beginConfiguration()
        captureSession.sessionPreset = .vga640x480

        guard
            let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
            let input = try? AVCaptureDeviceInput(device: camera),
            captureSession.canAddInput(input)
        else {
            captureSession.commitConfiguration()
            statusText = "Unable to access rear camera"
            return
        }
        captureSession.addInput(input)

        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: outputQueue)

        guard captureSession.canAddOutput(output) else {
            captureSession.commitConfiguration()
            statusText = "Unable to configure camera output"
            return
        }
        captureSession.addOutput(output)

        if let connection = output.connection(with: .video), connection.isVideoOrientationSupported {
            connection.videoOrientation = .portrait
        }

        captureSession.commitConfiguration()
        isConfigured = true

        outputQueue.async { [weak self] in
            self?.captureSession.startRunning()
        }
    }

    private func startStreaming() {
        guard canStream else {
            statusText = "Enter host and port"
            return
        }
        configureSessionIfNeeded()
        isStreaming = true
        statusText = "Starting..."
        lastError = nil
        sentTimestamps.removeAll()

        fpsTimer?.invalidate()
        fpsTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            guard let self else { return }
            let now = Date().timeIntervalSince1970
            self.sentTimestamps = self.sentTimestamps.filter { now - $0 <= 2.0 }
            self.senderFPS = Double(self.sentTimestamps.count) / 2.0
        }
    }

    private func stopStreaming(reason: String) {
        isStreaming = false
        statusText = reason
        senderFPS = 0
        sentTimestamps.removeAll()
        fpsTimer?.invalidate()
        fpsTimer = nil
    }

    private func upload(frameData: Data) async {
        let hostValue = host.trimmingCharacters(in: .whitespacesAndNewlines)
        let portValue = port.trimmingCharacters(in: .whitespacesAndNewlines)

        guard let url = URL(string: "http://\(hostValue):\(portValue)/frame") else {
            await MainActor.run {
                self.statusText = "Invalid host/port"
            }
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("image/jpeg", forHTTPHeaderField: "Content-Type")
        request.setValue(flipH ? "1" : "0", forHTTPHeaderField: "X-Flip-H")
        request.setValue(flipV ? "1" : "0", forHTTPHeaderField: "X-Flip-V")
        request.httpBody = frameData
        request.timeoutInterval = 2.0

        do {
            let (_, response) = try await urlSession.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse, (200 ... 299).contains(httpResponse.statusCode) else {
                throw URLError(.badServerResponse)
            }
            await MainActor.run {
                self.lastError = nil
                self.statusText = "Streaming"
                self.sentTimestamps.append(Date().timeIntervalSince1970)
            }
        } catch {
            await MainActor.run {
                self.lastError = error.localizedDescription
                self.statusText = "Stream error"
            }
        }
    }

    private func jpegData(from sampleBuffer: CMSampleBuffer) -> Data? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return nil
        }

        let image = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
            return nil
        }

        return UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.65)
    }
}

extension StreamingViewModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        Task { @MainActor [weak self] in
            guard let self, self.isStreaming else { return }

            var shouldUpload = false
            self.uploadStateQueue.sync {
                if !self.isUploading {
                    self.isUploading = true
                    shouldUpload = true
                }
            }

            guard shouldUpload else { return }
            defer {
                self.uploadStateQueue.sync {
                    self.isUploading = false
                }
            }

            guard let frameData = self.jpegData(from: sampleBuffer) else { return }
            await self.upload(frameData: frameData)
        }
    }
}
