import { GoogleGenAI } from "@google/genai";
import { SYSTEM_INSTRUCTION_KOREAN } from "../constants";

let ai: GoogleGenAI | null = null;

export const initializeGemini = (apiKey: string) => {
  ai = new GoogleGenAI({ apiKey });
};

export const sendMessageToGemini = async (
  message: string,
  history: { role: 'user' | 'model'; parts: { text: string }[] }[]
): Promise<string> => {
  if (!ai) {
    throw new Error("API Key not initialized");
  }

  try {
    const chat = ai.chats.create({
      model: 'gemini-2.5-flash',
      config: {
        systemInstruction: SYSTEM_INSTRUCTION_KOREAN,
      },
      history: history,
    });

    const result = await chat.sendMessage({ message });
    return result.text || "죄송합니다. 답변을 생성할 수 없습니다.";
  } catch (error) {
    console.error("Gemini API Error:", error);
    throw error;
  }
};
