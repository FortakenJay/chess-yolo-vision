import { CameraType, CameraView, useCameraPermissions } from 'expo-camera';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  AppState,
  Button,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from 'react-native';
import Slider from '@react-native-community/slider';
import { SafeAreaView } from 'react-native-safe-area-context';

type StreamStatus = 'idle' | 'streaming' | 'error';

const MIN_FPS = 5;
const MAX_FPS = 15;
const FPS_STEP = 1;
const MIN_QUALITY = 0.5;
const MAX_QUALITY = 1.0;
const QUALITY_STEP = 0.05;

export default function HomeScreen() {
  const cameraRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [isForeground, setIsForeground] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState<StreamStatus>('idle');
  const [statusText, setStatusText] = useState('Ready');
  const [displayFps, setDisplayFps] = useState(0);
  const [lastError, setLastError] = useState('');
  const [flipH, setFlipH] = useState(false);
  const [flipV, setFlipV] = useState(false);
  const [host, setHost] = useState('192.168.0.10');
  const [port, setPort] = useState('8080');
  const [targetFps, setTargetFps] = useState(12);
  const [jpegQuality, setJpegQuality] = useState(0.8);
  const sentTimestampsRef = useRef<number[]>([]);

  useEffect(() => {
    const sub = AppState.addEventListener('change', (nextState) => {
      setIsForeground(nextState === 'active');
    });
    return () => sub.remove();
  }, []);

  useEffect(() => {
    if (!isForeground && isStreaming) {
      setIsStreaming(false);
      setStatus('idle');
      setStatusText('Paused (app in background)');
    }
  }, [isForeground, isStreaming]);

  useEffect(() => {
    if (!isStreaming || !isForeground) {
      return;
    }

    const frameIntervalMs = Math.max(60, Math.round(1000 / targetFps));
    let stopped = false;

    const sendOneFrame = async () => {
      if (stopped || !cameraRef.current) {
        return;
      }
      try {
        const picture = await cameraRef.current.takePictureAsync({
          quality: jpegQuality,
          base64: false,
          skipProcessing: true,
        });
        if (!picture?.uri) {
          return;
        }

        const imageResponse = await fetch(picture.uri);
        const blob = await imageResponse.blob();
        const response = await fetch(`http://${host.trim()}:${port.trim()}/frame`, {
          method: 'POST',
          headers: {
            'Content-Type': 'image/jpeg',
            'X-Flip-H': flipH ? '1' : '0',
            'X-Flip-V': flipV ? '1' : '0',
          },
          body: blob,
        });

        if (!response.ok) {
          throw new Error(`Bridge returned ${response.status}`);
        }

        const now = Date.now();
        sentTimestampsRef.current.push(now);
        sentTimestampsRef.current = sentTimestampsRef.current.filter((t) => now - t <= 3000);
        setDisplayFps((sentTimestampsRef.current.length / 3.0) || 0);
        setStatus('streaming');
        setStatusText(`Streaming to ${host.trim()}:${port.trim()}`);
        setLastError('');
      } catch (error) {
        setStatus('error');
        setStatusText('Stream error');
        setLastError(error instanceof Error ? error.message : 'Unknown upload error');
      }
    };

    const timer = setInterval(() => {
      void sendOneFrame();
    }, frameIntervalMs);

    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [flipH, flipV, host, isForeground, isStreaming, jpegQuality, port, targetFps]);

  const canStream = useMemo(() => {
    return host.trim().length > 0 && port.trim().length > 0;
  }, [host, port]);

  if (Platform.OS === 'web') {
    return (
      <SafeAreaView style={styles.centered}>
        <Text style={styles.webText}>Use iOS or Android device (Expo Go) for camera streaming.</Text>
      </SafeAreaView>
    );
  }

  if (!permission) {
    return (
      <SafeAreaView style={styles.centered}>
        <Text style={styles.label}>Loading camera permission...</Text>
      </SafeAreaView>
    );
  }

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.centered}>
        <Text style={styles.label}>Camera permission is required.</Text>
        <View style={styles.permissionButton}>
          <Button title="Grant camera permission" onPress={requestPermission} />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.previewContainer}>
        <CameraView ref={cameraRef} style={styles.preview} facing={'back' as CameraType} />
        <View style={styles.overlay}>
          <Text style={styles.overlayText}>FPS: {displayFps.toFixed(1)}</Text>
          <Text style={styles.overlayText}>Status: {statusText}</Text>
        </View>
      </View>

      <ScrollView style={styles.controls} contentContainerStyle={styles.controlsContent}>
        <Text style={styles.sectionTitle}>Stream Controls</Text>
        <View style={styles.row}>
          <Text style={styles.label}>Flip Horizontal</Text>
          <Switch value={flipH} onValueChange={setFlipH} />
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>Flip Vertical</Text>
          <Switch value={flipV} onValueChange={setFlipV} />
        </View>
        <Pressable
          style={[styles.button, !canStream && styles.buttonDisabled]}
          disabled={!canStream}
          onPress={() => {
            const next = !isStreaming;
            setIsStreaming(next);
            if (!next) {
              setStatus('idle');
              setStatusText('Stopped');
            } else {
              setStatus('streaming');
              setStatusText('Starting stream...');
            }
          }}>
          <Text style={styles.buttonText}>{isStreaming ? 'Stop Streaming' : 'Start Streaming'}</Text>
        </Pressable>

        <Text style={styles.sectionTitle}>Bridge Settings</Text>
        <Text style={styles.label}>PC Host / LAN IP</Text>
        <TextInput
          style={styles.input}
          value={host}
          onChangeText={setHost}
          autoCapitalize="none"
          autoCorrect={false}
          placeholder="192.168.0.10"
          keyboardType="numbers-and-punctuation"
        />

        <Text style={styles.label}>Port</Text>
        <TextInput
          style={styles.input}
          value={port}
          onChangeText={setPort}
          autoCapitalize="none"
          autoCorrect={false}
          placeholder="8080"
          keyboardType="number-pad"
        />

        <Text style={styles.label}>Target FPS: {targetFps}</Text>
        <Slider
          minimumValue={MIN_FPS}
          maximumValue={MAX_FPS}
          step={FPS_STEP}
          value={targetFps}
          onValueChange={setTargetFps}
        />

        <Text style={styles.label}>JPEG Quality: {jpegQuality.toFixed(2)}</Text>
        <Slider
          minimumValue={MIN_QUALITY}
          maximumValue={MAX_QUALITY}
          step={QUALITY_STEP}
          value={jpegQuality}
          onValueChange={setJpegQuality}
        />

        {status === 'error' && lastError.length > 0 ? (
          <Text style={styles.errorText}>Error: {lastError}</Text>
        ) : null}
        <Text style={styles.hint}>
          Keep this app in foreground while streaming. Use bridge endpoint: http://&lt;pc-ip&gt;:&lt;port&gt;/frame
        </Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#111111',
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#111111',
  },
  previewContainer: {
    flex: 1,
    backgroundColor: '#000000',
  },
  preview: {
    flex: 1,
  },
  overlay: {
    position: 'absolute',
    top: 12,
    left: 12,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    borderRadius: 8,
    paddingVertical: 8,
    paddingHorizontal: 10,
    gap: 4,
  },
  overlayText: {
    color: '#00ff66',
    fontSize: 14,
    fontWeight: '700',
  },
  controls: {
    maxHeight: 360,
    backgroundColor: '#1a1a1a',
  },
  controlsContent: {
    padding: 14,
    gap: 10,
  },
  sectionTitle: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '700',
    marginTop: 6,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  label: {
    color: '#f2f2f2',
    fontSize: 14,
  },
  input: {
    backgroundColor: '#2a2a2a',
    borderWidth: 1,
    borderColor: '#3d3d3d',
    borderRadius: 8,
    color: '#ffffff',
    paddingHorizontal: 10,
    paddingVertical: 8,
  },
  button: {
    marginTop: 4,
    borderRadius: 8,
    backgroundColor: '#0f8f4a',
    paddingVertical: 12,
    alignItems: 'center',
  },
  buttonDisabled: {
    backgroundColor: '#606060',
  },
  buttonText: {
    color: '#ffffff',
    fontWeight: '700',
  },
  errorText: {
    color: '#ff8080',
    fontWeight: '700',
  },
  hint: {
    color: '#c5c5c5',
    fontSize: 12,
    marginBottom: 4,
  },
  webText: {
    color: '#f2f2f2',
    textAlign: 'center',
    fontSize: 16,
  },
  permissionButton: {
    marginTop: 12,
  },
});
