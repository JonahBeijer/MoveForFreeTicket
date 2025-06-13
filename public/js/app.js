import {
    initPoseDetection,
    startCollect,
    stopCollect,
    downloadData
} from "./poseDetection.js";

import {
    trainModelFromFile,
    predictPose,
    saveModel
} from "./trainModel.js";

// State
let trainedModel = null;
let squatCounter = 0;
let jackCounter = 0;
let lastSquatState = 'standing';
let lastJackState = 'rest';

// Keypoints indices (x, y)
const KEYPOINT_INDICES = {
    LEFT_HIP_Y: 47,
    RIGHT_HIP_Y: 49,
    LEFT_KNEE_Y: 51,
    RIGHT_KNEE_Y: 53,
    LEFT_WRIST_X: 30,
    RIGHT_WRIST_X: 32,
    LEFT_SHOULDER_X: 22,
    RIGHT_SHOULDER_X: 24
};

const THRESHOLDS = {
    SQUAT: { HIP_KNEE_DIFF: 0.12, TRANSITION_BUFFER: 0.05 },
    JUMPING_JACK: { WRIST_SPREAD_RATIO: 2.2, TRANSITION_BUFFER: 0.3 }
};

async function loadSavedModel() {
    return new Promise((resolve) => {
        if (!localStorage.getItem('poseModel')) {
            console.log('Geen models gevonden in localStorage');
            resolve(null);
            return;
        }

        console.log('Model gevonden, laden...');

        const model = ml5.neuralNetwork({
            task: 'classification',
            debug: true,
            inputs: 34,
            outputs: ['Squat', 'JumpingJack']
        });

        model.load('localstorage://poseModel', (err, loadedModel) => {
            if (err) {
                console.error('Laadfout:', err);
                resolve(null);
            } else {
                console.log('Model succesvol geladen:', loadedModel);
                resolve(loadedModel);
            }
        });
    });
}

function detectSquat(keypoints) {
    const hipAvgY = (keypoints[KEYPOINT_INDICES.LEFT_HIP_Y] + keypoints[KEYPOINT_INDICES.RIGHT_HIP_Y]) / 2;
    const kneeAvgY = (keypoints[KEYPOINT_INDICES.LEFT_KNEE_Y] + keypoints[KEYPOINT_INDICES.RIGHT_KNEE_Y]) / 2;
    const hipKneeDiff = hipAvgY - kneeAvgY;

    if (hipKneeDiff > THRESHOLDS.SQUAT.HIP_KNEE_DIFF && lastSquatState === 'standing') {
        lastSquatState = 'bottom';
    } else if (hipKneeDiff < -THRESHOLDS.SQUAT.TRANSITION_BUFFER && lastSquatState === 'bottom') {
        squatCounter++;
        document.getElementById('squatcount').textContent = squatCounter;
        lastSquatState = 'standing';
    }
}

function detectJumpingJack(keypoints) {
    const wristSpread = Math.abs(keypoints[KEYPOINT_INDICES.LEFT_WRIST_X] - keypoints[KEYPOINT_INDICES.RIGHT_WRIST_X]);
    const shoulderWidth = Math.abs(keypoints[KEYPOINT_INDICES.LEFT_SHOULDER_X] - keypoints[KEYPOINT_INDICES.RIGHT_SHOULDER_X]);
    const spreadRatio = wristSpread / (shoulderWidth || 0.1);

    if (spreadRatio > THRESHOLDS.JUMPING_JACK.WRIST_SPREAD_RATIO && lastJackState === 'rest') {
        lastJackState = 'extended';
    } else if (spreadRatio < (THRESHOLDS.JUMPING_JACK.WRIST_SPREAD_RATIO - THRESHOLDS.JUMPING_JACK.TRANSITION_BUFFER) && lastJackState === 'extended') {
        jackCounter++;
        document.getElementById('jumpcount').textContent = jackCounter;
        lastJackState = 'rest';
    }
}

async function handlePoseDetection(pose, keypoints) {
    detectSquat(keypoints);
    detectJumpingJack(keypoints);

    if (trainedModel) {
        try {
            // Normaliseer keypoints [0..1] evt.
            const maxVal = Math.max(...keypoints);
            const normKeypoints = keypoints.map(k => k / maxVal);
            const label = await predictPose(trainedModel, normKeypoints);
            // Je kunt hier je label gebruiken voor extra logica
        } catch (error) {
            console.error('Model error:', error);
        }
    }
}

async function initializeApp() {
    trainedModel = await loadSavedModel();
    initPoseDetection(handlePoseDetection);
}

window.addEventListener('DOMContentLoaded', initializeApp);

document.getElementById("train").onclick = async () => {
    try {
        const model = await trainModelFromFile("jsonFile");
        await saveModel(model);
        alert("Model getraind en gedownload. Sla op in /models/");
    } catch (e) {
        console.error(e);
        alert("Training mislukt.");
    }
};

document.getElementById("startSquat").onclick = () => startCollect("Squat");
document.getElementById("startJump").onclick = () => startCollect("JumpingJack");
document.getElementById("stopCollect").onclick = stopCollect;
document.getElementById("download").onclick = downloadData;
