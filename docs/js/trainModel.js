export async function trainModelFromFile(inputId) {
    const file = document.getElementById(inputId).files?.[0];
    if (!file) return alert("Upload een JSON-bestand.");
    let data;
    try { data = JSON.parse(await file.text()); }
    catch { return alert("Ongeldig JSON."); }

    const model = ml5.neuralNetwork({
        task: 'classification', debug: true, inputs: 66, outputs: ['Squat','JumpingJack'], learningRate: 0.01
    });
    data.forEach(s => model.addData(s.keypoints, { label: s.label }));
    await new Promise((res, rej) => model.train({epochs:50, batchSize:32}, err => err ? rej(err) : res()));
    return model;
}

export async function saveModel(model) {
    model.save('poseModel'); // download model.json + model.weights.bin
}


export async function trainModel(data) {
    const model = ml5.neuralNetwork({
        task: 'classification',
        debug: true,
        inputs: 66,
        outputs: 2,
        learningRate: 0.01,
    });

    // Data checks
    data.forEach(sample => {

        model.addData(sample.keypoints, { label: sample.label });
    });

    // Gecorrigeerde train functie
    await new Promise((resolve, reject) => {
        model.train(
            {
                epochs: 50,
                batchSize: 32,
                callback: (epoch, loss) => {  // <-- Correcte parameter naam
                    console.log(`Epoch ${epoch} - Loss: ${loss}`);
                }
            },
            (err) => {  // <-- Callback voor training voltooiing
                if(err) {
                    reject(err);
                } else {
                    resolve();
                }
            }
        );
    });

    return model;
}

function flattenKeypoints(keypoints) {
    if (!keypoints || keypoints.length === 0) {
        console.error("Ongeldige keypoints:", keypoints);
        return [];
    }
    return keypoints.reduce((acc, value, index) => {
        if (index % 2 === 0) { // X-coördinaten (even indexen)
            acc.push(value);
        } else if (index % 2 === 1) { // Y-coördinaten (oneven indexen)
            acc.push(value);
        }
        return acc;
    }, []);
}


function oneHotEncode(label, classes) {
    const encoding = Array(classes.length).fill(0);
    const index = classes.indexOf(label);
    if (index !== -1) encoding[index] = 1;
    return encoding;
}

export async function predictPose(model, keypoints) {
    if (!model || !keypoints || keypoints.length !== 66) {
        return "Unknown";
    }

    // Flatten de keypoints naar X, Y, Z
    const flattenedKeypoints = flattenKeypoints(keypoints);

    // Controleer of er geldige keypoints zijn
    if (flattenedKeypoints.length === 0) {
        return "Unknown";
    }

    // Normaliseren van de keypoints
    const max = Math.max(...flattenedKeypoints);
    if (max === 0) { // Voorkom deling door nul
        return "Unknown";
    }
    const normalizedKeypoints = flattenedKeypoints.map(value => value / max);

    return new Promise(resolve => {
        model.classify(normalizedKeypoints, (err, results) => {
            if (err || !results?.[0]?.label) {
                resolve("Unknown");
            } else {
                resolve(results[0].label);
            }
        });
    });
}