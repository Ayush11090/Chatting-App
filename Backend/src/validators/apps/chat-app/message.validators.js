import { body } from "express-validator";
import fetch from "node-fetch"; // Ensure node-fetch is installed: npm install node-fetch

// Function to call the Python API and get the predicted class
const classifyMessage = async (text) => {
  try {
    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });
    const data = await response.json();
    return data.predicted_class;
  } catch (error) {
    console.error("Error calling classification API:", error);
    throw new Error("Classification service is unavailable");
  }
};

const sendMessageValidator = () => {
  return [
    // Validate message content (text)
    body("content")
      .trim()
      .optional()
      .notEmpty()
      .withMessage("Content is required")
      .custom(async (content) => {
        const predictedClass = await classifyMessage(content);
        if (predictedClass !== 0) {
          throw new Error("This message is not displayed due to offensive content");
        }
        return true; // Pass validation if not offensive
      }),
  ];
};


export { sendMessageValidator };
