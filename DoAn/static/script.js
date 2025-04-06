async function checkSpam() {
    let emailContent = document.getElementById("emailInput").value.trim();
    let resultDiv = document.getElementById("result");
    let classification = document.getElementById("classification");
    let accuracy = document.getElementById("accuracy");

    if (emailContent === "") {
        alert(" Vui lòng nhập nội dung email!");
        return;
    }

    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email: emailContent })
        });

        if (!response.ok) {
            throw new Error('Không thể kết nối đến server');
        }

        const data = await response.json();

        const prediction = data.prediction;
        const confidence = data.accuracy.toFixed(2);

        if (prediction === "Spam mail") {
            classification.innerHTML = " <b>Đây là email spam!</b>";
            classification.style.color = "red";
        } else {
            classification.innerHTML = " <b>Đây là email hợp lệ (Ham)!</b>";
            classification.style.color = "green";
        }

        accuracy.innerHTML = ` Độ chính xác: <b>${confidence}%</b>`;
        resultDiv.classList.remove("hidden");
    } catch (error) {
        classification.innerHTML = " <b>Có lỗi xảy ra!</b>";
        classification.style.color = "orange";
        accuracy.innerHTML = `Thông báo lỗi: ${error.message}`;
        resultDiv.classList.remove("hidden");
    }
}
