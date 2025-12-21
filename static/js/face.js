const video = document.getElementById("video");
const resultBox = document.getElementById("result");

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream);

let referenceBlob = null;

// ---------- helpers ----------

function stats(arr) {
  const mean = arr.reduce((a,b)=>a+b,0)/arr.length;
  const std = Math.sqrt(arr.reduce((a,b)=>a+(b-mean)**2,0)/arr.length);
  const norm = Math.sqrt(arr.reduce((a,b)=>a+b*b,0));
  return { mean, std, norm };
}

async function captureFrame() {
  const c = document.createElement("canvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);
  return new Promise(r => c.toBlob(r, "image/jpeg"));
}

// ---------- buttons ----------

document.getElementById("captureRef").onclick = async () => {
  referenceBlob = await captureFrame();
  resultBox.innerHTML = "✅ Reference image captured";
};

document.getElementById("captureVerify").onclick = async () => {
  if (!referenceBlob) {
    resultBox.innerHTML = "❌ Capture reference first";
    return;
  }

  resultBox.innerHTML = "🔍 Verifying…";

  const liveBlob = await captureFrame();
  const form = new FormData();
  form.append("reference", referenceBlob, "ref.jpg");
  form.append("live", liveBlob, "live.jpg");

  const res = await fetch("/api/verify-face", {
    method: "POST",
    body: form
  });

  const data = await res.json();

  const r = data.embeddings.reference_embedding;
  const l = data.embeddings.live_embedding;
  const rs = stats(r);
  const ls = stats(l);

  resultBox.innerHTML = `
<div class="glass p-4 rounded-xl">
  <h3 class="text-xl font-bold">
    ${data.verified ? "✅ VERIFIED" : "❌ NOT VERIFIED"}
  </h3>
  <p>Score: ${data.scores.face_similarity.toFixed(4)}</p>
  <p>Confidence: ${data.confidence_bucket}</p>
</div>

<details class="glass p-4 rounded-xl">
  <summary class="font-semibold cursor-pointer text-indigo-400">
    Embeddings
  </summary>

  <div class="grid md:grid-cols-2 gap-4 mt-4">
    <div>
      <p>Ref norm: ${rs.norm.toFixed(3)}</p>
      <details><summary>Raw</summary>
        <pre class="text-xs max-h-48 overflow-auto">
${JSON.stringify(r, null, 2)}
        </pre>
      </details>
    </div>

    <div>
      <p>Live norm: ${ls.norm.toFixed(3)}</p>
      <details><summary>Raw</summary>
        <pre class="text-xs max-h-48 overflow-auto">
${JSON.stringify(l, null, 2)}
        </pre>
      </details>
    </div>
  </div>
</details>

<details class="glass p-4 rounded-xl">
  <summary class="font-semibold cursor-pointer text-indigo-400">
    Full Raw JSON
  </summary>
  <pre class="text-xs max-h-96 overflow-auto">
${JSON.stringify(data, null, 2)}
  </pre>
</details>
`;
};
