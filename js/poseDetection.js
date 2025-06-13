import {
    PoseLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

export let poseLandmarker;
export let video;
export let canvas;
export let ctx;
export let drawUtils;
export let poseData = [];
export let collecting = false;
export let collectLabel = "";

export async function initPoseDetection(onPoseDetected) {
    video = document.getElementById("webcam");
    canvas = document.getElementById("output_canvas");
    ctx = canvas.getContext("2d");
    drawUtils = new DrawingUtils(ctx);

    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1
    });

    await startCamera();
    detectPoseLoop(onPoseDetected);
}

export function startCollect(label) {
    collectLabel = label;
    collecting = true;
}

export function stopCollect() {
    collecting = false;
}

export function downloadData() {
    const blob = new Blob([JSON.stringify(poseData, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "pose_dataset.json";
    a.click();
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;

        return new Promise(resolve => {
            video.onloadeddata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                resolve();
            };
        });
    } catch (error) {
        console.error("Error with webcam:", error);
        alert("Er is een probleem met het openen van de webcam.");
    }
}


async function detectPoseLoop(onPoseDetected) {
    if (!poseLandmarker) return;

    const results = await poseLandmarker.detectForVideo(video, performance.now());
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.landmarks.length > 0) {
        for (let pose of results.landmarks) {
            if (pose.length !== 33) continue;

            drawUtils.drawConnectors(pose, PoseLandmarker.POSE_CONNECTIONS, {
                color: "#00FF00",
                lineWidth: 4
            });

            drawUtils.drawLandmarks(pose, {
                radius: 4,
                color: "#FF0000"
            });

            const keypoints = pose.map(k => [k.x, k.y]).flat();

            // ✅ Check of onderlichaam zichtbaar is
            const lowerVisible = isLowerBodyVisible(pose);

            if (collecting) {
                poseData.push({ keypoints, label: collectLabel });
            }

            if (typeof onPoseDetected === "function") {
                onPoseDetected(pose, keypoints, lowerVisible);
            }
        }
    }

    requestAnimationFrame(() => detectPoseLoop(onPoseDetected));
}

// ✅ NIEUWE FUNCTIE: check zichtbaarheid onderlichaam
function isLowerBodyVisible(pose) {
    const MIN_VISIBILITY = 0.6; // Je kunt dit aanpassen

    const landmarksToCheck = [
        23, // left hip
        24, // right hip
        25, // left knee
        26, // right knee
        27, // left ankle
        28  // right ankle
    ];

    return landmarksToCheck.every(i => pose[i].visibility > MIN_VISIBILITY);
}

