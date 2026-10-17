import SwiftUI

@main
struct PhoneStreamerApp: App {
    @StateObject private var viewModel = StreamingViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(viewModel)
        }
    }
}
