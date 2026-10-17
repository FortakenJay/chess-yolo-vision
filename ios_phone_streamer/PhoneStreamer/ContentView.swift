import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var viewModel: StreamingViewModel
    @State private var showingSettings = false
    @FocusState private var keyboardFocused: Bool

    var body: some View {
        ZStack {
            CameraPreviewView(session: viewModel.captureSession)
                .ignoresSafeArea()
                .onTapGesture {
                    keyboardFocused = false
                }

            VStack(spacing: 0) {
                topBar
                Spacer()
                bottomBar
            }
        }
        .preferredColorScheme(.dark)
        .sheet(isPresented: $showingSettings) {
            SettingsSheet()
                .environmentObject(viewModel)
        }
        .onAppear { viewModel.onAppear() }
        .onDisappear { viewModel.onDisappear() }
    }

    private var topBar: some View {
        HStack(alignment: .top) {
            statusChip
            Spacer()
            settingsButton
        }
        .padding(.horizontal, 16)
        .padding(.top, 8)
    }

    private var statusChip: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(String(format: "%.1f fps", viewModel.senderFPS))
                .font(.system(size: 22, weight: .heavy, design: .rounded))
                .foregroundColor(.green)
            Text(viewModel.statusText)
                .font(.caption2)
                .foregroundColor(.white.opacity(0.85))
                .lineLimit(1)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.black.opacity(0.55))
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private var settingsButton: some View {
        Button {
            showingSettings = true
        } label: {
            Image(systemName: "gearshape.fill")
                .font(.system(size: 20, weight: .semibold))
                .foregroundColor(.white)
                .frame(width: 44, height: 44)
                .background(.black.opacity(0.55))
                .clipShape(Circle())
        }
        .accessibilityLabel("Open settings")
    }

    private var bottomBar: some View {
        VStack(spacing: 14) {
            HStack(spacing: 10) {
                FlipChip(title: "Flip H", isOn: $viewModel.flipH)
                FlipChip(title: "Flip V", isOn: $viewModel.flipV)
            }

            Button {
                viewModel.toggleStreaming()
            } label: {
                HStack(spacing: 10) {
                    Circle()
                        .fill(viewModel.isStreaming ? Color.red : Color.white)
                        .frame(width: 14, height: 14)
                    Text(viewModel.isStreaming ? "Stop" : "Start Streaming")
                        .font(.system(size: 18, weight: .bold, design: .rounded))
                }
                .frame(maxWidth: .infinity, minHeight: 60)
                .background(viewModel.isStreaming ? Color.red.opacity(0.85) : Color.green.opacity(0.85))
                .foregroundColor(.white)
                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            }
            .disabled(!viewModel.canStream)
            .opacity(viewModel.canStream ? 1.0 : 0.5)

            if let error = viewModel.lastError, !error.isEmpty {
                Text(error)
                    .font(.caption)
                    .foregroundColor(.red)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(16)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
        .padding(.horizontal, 12)
        .padding(.bottom, 8)
    }
}

private struct FlipChip: View {
    let title: String
    @Binding var isOn: Bool

    var body: some View {
        Button {
            isOn.toggle()
        } label: {
            Text("\(title): \(isOn ? "On" : "Off")")
                .font(.system(size: 14, weight: .semibold))
                .frame(maxWidth: .infinity, minHeight: 40)
                .foregroundColor(.white)
                .background(isOn ? Color.green.opacity(0.6) : Color.white.opacity(0.12))
                .clipShape(Capsule())
        }
    }
}

private struct SettingsSheet: View {
    @EnvironmentObject private var viewModel: StreamingViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            Form {
                Section {
                    LabeledRow(label: "PC IP") {
                        TextField("192.168.0.10", text: $viewModel.host)
                            .keyboardType(.numbersAndPunctuation)
                            .textInputAutocapitalization(.never)
                            .disableAutocorrection(true)
                    }
                    LabeledRow(label: "Port") {
                        TextField("8080", text: $viewModel.port)
                            .keyboardType(.numberPad)
                    }
                } header: {
                    Text("Bridge Endpoint")
                } footer: {
                    Text("Phone uploads JPEG frames to http://<PC IP>:<port>/frame on your local Wi-Fi.")
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

private struct LabeledRow<Content: View>: View {
    let label: String
    let content: () -> Content

    var body: some View {
        HStack {
            Text(label)
                .frame(width: 80, alignment: .leading)
                .foregroundColor(.secondary)
            content()
        }
    }
}
