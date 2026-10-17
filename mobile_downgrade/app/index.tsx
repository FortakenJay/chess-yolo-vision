import Constants, { ExecutionEnvironment } from "expo-constants";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AppState,
  Button,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from "react-native-vision-camera";
import { type BridgeConfig, uploadFrameFromPath } from "@/lib/bridge";

type StreamStatus = "idle" | "streaming" | "error";

const isExpoGo =
  Constants.executionEnvironment === ExecutionEnvironment.StoreClient;

export default function HomeScreen() {
  const device = useCameraDevice("back");
  const { hasPermission, requestPermission } = useCameraPermission();
  const cameraRef = useRef<Camera>(null);

  const [isForeground, setIsForeground] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isActive, setIsActive] = useState(true);
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [statusText, setStatusText] = useState("Ready");
  const [displayFps, setDisplayFps] = useState(0);
  const [lastError, setLastError] = useState("");
  const [flipH, setFlipH] = useState(false);
  const [flipV, setFlipV] = useState(false);
  const [host, setHost] = useState("192.168.0.10");
  const [port, setPort] = useState("8080");
  const [showSettings, setShowSettings] = useState(false);

  const configRef = useRef<BridgeConfig>({ host, port, flipH, flipV });
  const isStreamingRef = useRef(false);
  const uploadingRef = useRef(false);
  const sentTimestampsRef = useRef<number[]>([]);

  useEffect(() => {
    configRef.current = { host, port, flipH, flipV };
  }, [host, port, flipH, flipV]);

  useEffect(() => {
    isStreamingRef.current = isStreaming;
  }, [isStreaming]);

  useEffect(() => {
    const sub = AppState.addEventListener("change", (nextState) => {
      const active = nextState === "active";
      setIsForeground(active);
      setIsActive(active);
      if (!active && isStreamingRef.current) {
        setIsStreaming(false);
        setStatus("idle");
        setStatusText("Paused (background)");
      }
    });
    return () => sub.remove();
  }, []);

  const canStream = useMemo(
    () => host.trim().length > 0 && port.trim().length > 0,
    [host, port],
  );

  const captureAndUpload = useCallback(async () => {
    const camera = cameraRef.current;
    if (!camera) return;

    const snapshot = await camera.takeSnapshot({
      quality: 70,
    });
    if (!snapshot?.path) return;

    await uploadFrameFromPath(configRef.current, snapshot.path);
    const now = Date.now();
    sentTimestampsRef.current.push(now);
    sentTimestampsRef.current = sentTimestampsRef.current.filter(
      (t) => now - t <= 2000,
    );
    setDisplayFps(sentTimestampsRef.current.length / 2);
    setStatus("streaming");
    setStatusText("Streaming");
    setLastError("");
  }, []);

  useEffect(() => {
    if (!isStreaming || !isForeground) return;

    let stopped = false;
    const fpsTicker = setInterval(() => {
      const now = Date.now();
      sentTimestampsRef.current = sentTimestampsRef.current.filter(
        (t) => now - t <= 2000,
      );
      setDisplayFps(sentTimestampsRef.current.length / 2);
    }, 500);

    const loop = async () => {
      while (!stopped && isStreamingRef.current) {
        if (!uploadingRef.current && cameraRef.current) {
          uploadingRef.current = true;
          try {
            await captureAndUpload();
          } catch (error) {
            setStatus("error");
            setStatusText("Stream error");
            setLastError(
              error instanceof Error ? error.message : "Upload failed",
            );
            await new Promise((r) => setTimeout(r, 250));
          } finally {
            uploadingRef.current = false;
          }
        }
        await new Promise((r) => setTimeout(r, 0));
      }
    };

    void loop();

    return () => {
      stopped = true;
      clearInterval(fpsTicker);
    };
  }, [captureAndUpload, isForeground, isStreaming]);

  if (Platform.OS === "web") {
    return (
      <SafeAreaView style={styles.centered}>
        <Text style={styles.label}>
          Use a dev build on iOS/Android. For browser streaming use web_streamer/.
        </Text>
      </SafeAreaView>
    );
  }

  if (isExpoGo) {
    return (
      <SafeAreaView style={styles.centered}>
        <Text style={styles.title}>Expo Go not supported</Text>
        <Text style={styles.label}>
          Vision Camera requires a custom dev client. Run: eas build --profile
          development --platform ios
        </Text>
        <Text style={styles.hint}>
          Or use web_streamer in Safari (no build).
        </Text>
      </SafeAreaView>
    );
  }

  if (!hasPermission) {
    return (
      <SafeAreaView style={styles.centered}>
        <Text style={styles.label}>Camera permission is required.</Text>
        <View style={styles.permissionButton}>
          <Button title="Grant camera permission" onPress={requestPermission} />
        </View>
      </SafeAreaView>
    );
  }

  if (!device) {
    return (
      <SafeAreaView style={styles.centered}>
        <Text style={styles.label}>No rear camera found on this device.</Text>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.previewContainer}>
        <Camera
          ref={cameraRef}
          style={styles.preview}
          device={device}
          isActive={isActive}
          photo
          video={false}
          audio={false}
        />
        <View style={styles.overlay}>
          <Text style={styles.overlayFps}>{displayFps.toFixed(1)} fps</Text>
          <Text style={styles.overlayStatus}>{statusText}</Text>
          <Text style={styles.overlayBadge}>Dev client</Text>
        </View>
        <Pressable style={styles.gearButton} onPress={() => setShowSettings(true)}>
          <Text style={styles.gearText}>⚙</Text>
        </Pressable>
      </View>

      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : undefined}
        keyboardVerticalOffset={16}>
        <View style={styles.controlsShell}>
          <View style={styles.quickRow}>
            <Pressable
              style={[styles.pillButton, flipH && styles.pillButtonActive]}
              onPress={() => setFlipH((v) => !v)}>
              <Text style={styles.pillText}>Flip H {flipH ? "On" : "Off"}</Text>
            </Pressable>
            <Pressable
              style={[styles.pillButton, flipV && styles.pillButtonActive]}
              onPress={() => setFlipV((v) => !v)}>
              <Text style={styles.pillText}>Flip V {flipV ? "On" : "Off"}</Text>
            </Pressable>
          </View>

          <Pressable
            style={[
              styles.streamButton,
              !canStream && styles.buttonDisabled,
              isStreaming && styles.streamButtonStop,
            ]}
            disabled={!canStream}
            onPress={() => {
              const next = !isStreaming;
              setIsStreaming(next);
              if (!next) {
                setStatus("idle");
                setStatusText("Stopped");
                setDisplayFps(0);
              } else {
                setStatus("streaming");
                setStatusText("Starting...");
              }
            }}>
            <Text style={styles.buttonText}>
              {isStreaming ? "Stop" : "Start Streaming"}
            </Text>
          </Pressable>

          {status === "error" && lastError.length > 0 ? (
            <Text style={styles.errorText}>Error: {lastError}</Text>
          ) : null}
        </View>
      </KeyboardAvoidingView>

      <SettingsModal
        visible={showSettings}
        host={host}
        port={port}
        onHost={setHost}
        onPort={setPort}
        onClose={() => setShowSettings(false)}
      />
    </SafeAreaView>
  );
}

function SettingsModal({
  visible,
  host,
  port,
  onHost,
  onPort,
  onClose,
}: {
  visible: boolean;
  host: string;
  port: string;
  onHost: (v: string) => void;
  onPort: (v: string) => void;
  onClose: () => void;
}) {
  return (
    <Modal visible={visible} animationType="slide" transparent onRequestClose={onClose}>
      <View style={styles.modalBackdrop}>
        <View style={styles.modalSheet}>
          <Text style={styles.modalTitle}>Bridge settings</Text>
          <Text style={styles.modalLabel}>PC LAN IP</Text>
          <TextInput
            style={styles.input}
            value={host}
            onChangeText={onHost}
            autoCapitalize="none"
            autoCorrect={false}
            placeholder="192.168.0.10"
            placeholderTextColor="#777"
            keyboardType="numbers-and-punctuation"
          />
          <Text style={styles.modalLabel}>Port</Text>
          <TextInput
            style={styles.input}
            value={port}
            onChangeText={onPort}
            autoCapitalize="none"
            autoCorrect={false}
            placeholder="8080"
            placeholderTextColor="#777"
            keyboardType="number-pad"
          />
          <Text style={styles.hint}>
            Uses Vision Camera snapshots (dev client). Bridge: POST /frame → MJPEG /stream.
          </Text>
          <Pressable style={styles.doneButton} onPress={onClose}>
            <Text style={styles.buttonText}>Done</Text>
          </Pressable>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#111111" },
  centered: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
    backgroundColor: "#111111",
  },
  previewContainer: { flex: 1, backgroundColor: "#000" },
  preview: { flex: 1 },
  overlay: {
    position: "absolute",
    top: 14,
    left: 14,
    backgroundColor: "rgba(0,0,0,0.55)",
    borderRadius: 10,
    paddingVertical: 8,
    paddingHorizontal: 12,
    gap: 2,
  },
  overlayFps: { color: "#00ff66", fontSize: 22, fontWeight: "800" },
  overlayStatus: { color: "#ddd", fontSize: 12 },
  overlayBadge: { color: "#8ab4ff", fontSize: 10, marginTop: 2 },
  gearButton: {
    position: "absolute",
    top: 14,
    right: 14,
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: "rgba(0,0,0,0.55)",
    alignItems: "center",
    justifyContent: "center",
  },
  gearText: { color: "#fff", fontSize: 20 },
  title: { color: "#fff", fontSize: 18, fontWeight: "700", marginBottom: 12 },
  label: { color: "#f2f2f2", fontSize: 14, textAlign: "center" },
  hint: { color: "#999", fontSize: 12, marginVertical: 8 },
  input: {
    backgroundColor: "#2a2a2a",
    borderWidth: 1,
    borderColor: "#3d3d3d",
    borderRadius: 10,
    color: "#fff",
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 16,
    marginBottom: 10,
  },
  buttonDisabled: { backgroundColor: "#606060" },
  buttonText: { color: "#fff", fontWeight: "800", fontSize: 16 },
  errorText: { color: "#ff8080", marginTop: 8, textAlign: "center" },
  permissionButton: { marginTop: 12 },
  controlsShell: {
    backgroundColor: "#171717",
    borderTopLeftRadius: 18,
    borderTopRightRadius: 18,
    paddingHorizontal: 14,
    paddingTop: 12,
    paddingBottom: 14,
    gap: 10,
  },
  quickRow: { flexDirection: "row", gap: 8 },
  pillButton: {
    flex: 1,
    backgroundColor: "#2a2a2a",
    borderWidth: 1,
    borderColor: "#3d3d3d",
    borderRadius: 999,
    paddingVertical: 12,
    alignItems: "center",
  },
  pillButtonActive: { backgroundColor: "#2e5f43", borderColor: "#4da66f" },
  pillText: { color: "#fff", fontWeight: "700", fontSize: 14 },
  streamButton: {
    borderRadius: 14,
    backgroundColor: "#0f8f4a",
    paddingVertical: 16,
    alignItems: "center",
  },
  streamButtonStop: { backgroundColor: "#a23737" },
  modalBackdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.5)",
    justifyContent: "flex-end",
  },
  modalSheet: {
    backgroundColor: "#1a1a1a",
    borderTopLeftRadius: 18,
    borderTopRightRadius: 18,
    padding: 20,
    paddingBottom: 32,
  },
  modalTitle: { color: "#fff", fontSize: 18, fontWeight: "700", marginBottom: 12 },
  modalLabel: { color: "#aaa", fontSize: 13, marginBottom: 4 },
  doneButton: {
    marginTop: 8,
    backgroundColor: "#2563eb",
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
  },
});
