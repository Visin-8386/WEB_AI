// ...JS code extracted from <script> in index.html...
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

// Hàm khởi tạo canvas với nền mặc định (mặc định là nền đen)
function initCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}
initCanvas();

// Các cài đặt ban đầu cho nét vẽ
let brushColor = document.getElementById("colorPicker").value;
let brushThickness = document.getElementById("thickness").value;
ctx.lineWidth = brushThickness;
ctx.lineCap = "round";
ctx.strokeStyle = brushColor;

// Cập nhật độ dày nét bút theo thanh trượt
document.getElementById("thickness").addEventListener("input", function () {
    brushThickness = this.value;
    ctx.lineWidth = brushThickness;
    document.getElementById("thicknessValue").innerText = brushThickness;
});

// Cập nhật màu nét bút theo lựa chọn của người dùng
document.getElementById("colorPicker").addEventListener("input", function () {
    brushColor = this.value;
    ctx.strokeStyle = brushColor;
});

// Cập nhật màu nền khung vẽ theo lựa chọn của người dùng
document.getElementById("bgColorPicker").addEventListener("input", function () {
    const bgColor = this.value;
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
});

let isDrawing = false;
canvas.addEventListener("mousedown", (e) => {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});
canvas.addEventListener("mousemove", (e) => {
    if (isDrawing) {
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    }
});
canvas.addEventListener("mouseup", () => { isDrawing = false; });
canvas.addEventListener("mouseout", () => { isDrawing = false; });

function clearCanvas() {
    initCanvas();
    document.getElementById("result").innerText = "Vẽ số và nhấn 'Dự đoán'";
    document.getElementById("digits-overview").innerHTML = "";
    document.getElementById("steps-details").innerHTML = "";
    // Ẩn cảnh báo khi xóa canvas
    document.getElementById("warning-section").style.display = "none";
}

// Hàm chuyển tab
function showTab(tabName) {
    const tabs = document.querySelectorAll('.tab-content');
    const buttons = document.querySelectorAll('.tab-button');

    tabs.forEach(tab => tab.style.display = 'none');
    buttons.forEach(btn => btn.classList.remove('active'));

    document.getElementById(`${tabName}-tab`).style.display = 'block';
    document.querySelector(`.tab-button[onclick="showTab('${tabName}')"]`).classList.add('active');
}

async function predict() {
    let dataURL = canvas.toDataURL("image/png");
    document.getElementById("result").innerHTML = `<span style="color:#4fc3f7;">Đang xử lý...</span>`;
    try {
        let response = await fetch("/predict_multiple", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: dataURL })
        });
        if (!response.ok) {
            throw new Error("Lỗi server hoặc không nhận được phản hồi!");
        }
        let result = await response.json();
        console.log("Kết quả trả về:", result);
        if (result.error) {
            document.getElementById("result").innerHTML = `<span style="color:red;">Lỗi: ${result.error}</span>`;
            document.getElementById("warning-section").style.display = "none";
        } else {
            // Phân loại số dựa trên độ tin cậy
            const digitsOverview = document.getElementById("digits-overview");
            digitsOverview.innerHTML = "";
            const warningDigits = [];
            const normalDigits = [];
            const normalIndexes = [];
            const warningIndexes = [];
            result.confidence.forEach((conf, idx) => {
                if (conf < 0.55) {
                    warningDigits.push({
                        digit: result.digits[idx],
                        confidence: conf,
                        img: result.steps.digits_resized[idx]
                    });
                    warningIndexes.push(idx);
                } else {
                    normalDigits.push({
                        digit: result.digits[idx],
                        confidence: conf,
                        img: result.steps.digits_resized[idx]
                    });
                    normalIndexes.push(idx);
                }
            });

            // Hiển thị kết quả dự đoán chỉ với các số có độ tin cậy >= 55%
            if (normalDigits.length > 0) {
                document.getElementById("result").innerText =
                    `Dự đoán: ${normalDigits.map(d => d.digit).join("")}`;
            } else {
                document.getElementById("result").innerText = "Không có số nào đủ độ tin cậy!";
            }

            // Hiển thị tổng quan (chỉ các số đủ tin cậy)
            normalDigits.forEach((item, index) => {
                const digitContainer = document.createElement("div");
                digitContainer.className = "digit-item";
                const img = document.createElement("img");
                img.src = "data:image/png;base64," + item.img;
                img.className = "digit-image";
                const label = document.createElement("div");
                label.innerText = `Số ${item.digit} (Độ tin cậy: ${(item.confidence * 100).toFixed(2)}%)`;
                digitContainer.appendChild(img);
                digitContainer.appendChild(label);
                digitsOverview.appendChild(digitContainer);
            });

            // Hiển thị cảnh báo nếu có số độ tin cậy thấp
            const warningSection = document.getElementById("warning-section");
            const warningDigitsDiv = document.getElementById("warning-digits");
            if (warningDigits.length > 0) {
                warningSection.style.display = "block";
                warningSection.style.background = "rgba(255,0,0,0.18)";
                warningSection.style.boxShadow = "0 0 20px #ff4b5c";
                warningDigitsDiv.innerHTML = "";
                warningDigits.forEach((item, idx) => {
                    const digitContainer = document.createElement("div");
                    digitContainer.className = "digit-item";
                    digitContainer.style.display = "inline-block";
                    digitContainer.style.marginRight = "20px";
                    const img = document.createElement("img");
                    img.src = "data:image/png;base64," + item.img;
                    img.className = "digit-image";
                    const label = document.createElement("div");
                    label.innerText = `Số ${item.digit} (Độ tin cậy: ${(item.confidence * 100).toFixed(2)}%)`;
                    digitContainer.appendChild(img);
                    digitContainer.appendChild(label);
                    warningDigitsDiv.appendChild(digitContainer);
                });
            } else {
                warningSection.style.display = "none";
            }

            // Hiển thị chi tiết các bước xử lý
            const stepsDetails = document.getElementById("steps-details");
            stepsDetails.innerHTML = "";
            // Bước 1: Ảnh gốc
            addStepSection(stepsDetails, "Bước 1: Ảnh gốc",
                "Ảnh đầu vào được chuyển sang grayscale (thang độ xám).",
                "data:image/png;base64," + result.steps.gray);
            // Bước 2: Ảnh làm mờ
            addStepSection(stepsDetails, "Bước 2: Ảnh làm mờ",
                "Áp dụng Gaussian blur để giảm nhiễu.",
                "data:image/png;base64," + result.steps.blurred);
            // Bước 3: Nền được xử lý
            addStepSection(stepsDetails, "Bước 3: Nền được xử lý",
                "Nền được xử lý để đảm bảo nền đen và chữ trắng.",
                "data:image/png;base64," + result.steps.optimized_background);
            // Bước 4: Tìm contours
            if (result.steps.contours) {
                addStepSection(stepsDetails, "Bước 4: Tìm contours",
                    "Tìm các đường viền (contours) của các đối tượng trong ảnh.",
                    "data:image/png;base64," + result.steps.contours);
            }
            // Bước 5: Xác định bounding boxes
            if (result.steps.boxes) {
                addStepSection(stepsDetails, "Bước 5: Xác định bounding boxes",
                    "Tạo các hộp bao quanh cho mỗi contour đủ lớn.",
                    "data:image/png;base64," + result.steps.boxes);
            }
            // Bước 6: Sắp xếp bounding boxes
            if (result.steps.sorted_boxes) {
                addStepSection(stepsDetails, "Bước 6: Sắp xếp bounding boxes",
                    "Sắp xếp các bounding boxes theo thứ tự từ trái sang phải và từ trên xuống dưới.",
                    "data:image/png;base64," + result.steps.sorted_boxes);
            }
            // Bước 7: Tăng độ dày nét vẽ và làm bự chữ số
            result.steps.thickened_digits.forEach((base64Str, index) => {
                if (result.steps.is_thickened_digits[index]) {
                    addStepSection(stepsDetails, `Bước 7.${index + 1}: Tăng độ dày nét vẽ`,
                        "Tăng độ dày nét vẽ và làm bự chữ số nếu kích thước quá nhỏ.",
                        "data:image/png;base64," + base64Str);
                }
            });
            // Bước 8: Cắt và resize các chữ số
            const digitStepContainer = document.createElement("div");
            digitStepContainer.className = "step-container";
            const digitStepTitle = document.createElement("div");
            digitStepTitle.className = "step-title";
            digitStepTitle.innerText = "Bước 8: Cắt và resize các chữ số";
            const digitStepDesc = document.createElement("div");
            digitStepDesc.innerText = "Mỗi chữ số được cắt từ ảnh gốc và resize về kích thước 28x28 pixel.";
            const digitGrid = document.createElement("div");
            digitGrid.className = "digit-container";
            result.steps.digits_raw.forEach((rawDigit, index) => {
                const digitPair = document.createElement("div");
                digitPair.className = "digit-item";
                digitPair.style.marginRight = "20px";
                let rawImgSrc;
                if (result.steps.is_thickened_digits[index]) {
                    rawImgSrc = "data:image/png;base64," + result.steps.thickened_digits[index];
                } else {
                    rawImgSrc = "data:image/png;base64," + rawDigit;
                }
                const rawImg = document.createElement("img");
                rawImg.src = rawImgSrc;
                rawImg.className = "digit-image";
                rawImg.style.marginBottom = "5px";
                const resizedImg = document.createElement("img");
                resizedImg.src = "data:image/png;base64," + result.steps.digits_resized[index];
                resizedImg.className = "digit-image";
                const stt = document.createElement("div");
                stt.style.fontWeight = "bold";
                stt.style.marginBottom = "3px";
                stt.innerText = `#${index}`;
                const label = document.createElement("div");
                label.innerText = `Dự đoán: ${result.digits[index]} (Độ tin cậy: ${(result.confidence[index] * 100).toFixed(2)}%)`;
                digitPair.appendChild(stt);
                digitPair.appendChild(document.createTextNode("Gốc:"));
                digitPair.appendChild(rawImg);
                digitPair.appendChild(document.createTextNode("Resize:"));
                digitPair.appendChild(resizedImg);
                digitPair.appendChild(label);
                digitGrid.appendChild(digitPair);
            });
            digitStepContainer.appendChild(digitStepTitle);
            digitStepContainer.appendChild(digitStepDesc);
            digitStepContainer.appendChild(digitGrid);
            stepsDetails.appendChild(digitStepContainer);
            // Hiển thị tab tổng quan mặc định
            showTab('overview');
        }
    } catch (err) {
        document.getElementById("result").innerHTML = `<span style='color:red;'>Lỗi kết nối hoặc server: ${err.message}</span>`;
        document.getElementById("warning-section").style.display = "none";
        console.error("Lỗi khi gọi API hoặc xử lý kết quả:", err);
    }
}

function addStepSection(container, title, description, imageUrl) {
    const stepContainer = document.createElement("div");
    stepContainer.className = "step-container";

    const stepTitle = document.createElement("div");
    stepTitle.className = "step-title";
    stepTitle.innerText = title;

    const stepDesc = document.createElement("div");
    stepDesc.className = "step-description";
    stepDesc.innerText = description;

    const stepImage = document.createElement("img");
    stepImage.src = imageUrl;
    stepImage.className = "step-image";

    // Đảm bảo thứ tự: tiêu đề -> mô tả -> ảnh
    stepContainer.appendChild(stepTitle);
    stepContainer.appendChild(stepDesc);
    stepContainer.appendChild(stepImage);
    container.appendChild(stepContainer);
}

function openDialog() {
    document.getElementById("guideDialog").showModal();
}

function closeDialog() {
    document.getElementById("guideDialog").close();
}

function openImageDialog() {
    document.getElementById("imageInput").click();
}

document.getElementById("imageInput").addEventListener("change", function (event) {
    const files = event.target.files;
    if (files && files.length > 0) {
        // Hiển thị ảnh đầu tiên lên canvas (các ảnh còn lại sẽ được xử lý khi upload/predict)
        const file = files[0];
        const reader = new FileReader();
        reader.onload = function (e) {
            const img = new Image();
            img.onload = function () {
                // Tính toán tỷ lệ để vẽ ảnh vừa khít canvas
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0, img.width, img.height);
            }
            img.src = e.target.result;
        }
        reader.readAsDataURL(file);
    }
});
// ...existing code...
