import { GoogleGenAI, GenerateContentParameters, Part, Chat } from "@google/genai";
import { Message, Author, GroundingSource } from "../types";
import { SYSTEM_PROMPT } from "../constants";

let ai: GoogleGenAI | null = null;

const getAIClient = () => {
    if (ai) return ai;

    const apiKey = process.env.API_KEY || localStorage.getItem("GEMINI_API_KEY");

    if (!apiKey) {
        throw new Error("API Key not found. Please ensure it is set in environment or localStorage.");
    }

    ai = new GoogleGenAI({ apiKey });
    return ai;
};

const fileToGenerativePart = async (file: File): Promise<Part> => {
    const base64EncodedDataPromise = new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
        reader.readAsDataURL(file);
    });
    return {
        inlineData: {
            data: await base64EncodedDataPromise,
            mimeType: file.type,
        },
    };
};


function buildGeminiHistory(messages: Message[]) {
    // Note: Gemini history alternates between 'user' and 'model'
    // The first message in Gemini history must be from a 'user'.
    const history = [...messages];
    if (history.length > 0 && history[0].author === Author.AI) {
        history.shift();
    }
    return history.map(msg => ({
        role: msg.author === Author.USER ? 'user' : 'model',
        parts: [{ text: msg.content }]
    }));
}

export async function* getChatResponseStream(history: Message[], newMessage: string): AsyncGenerator<{ text: string; sources: GroundingSource[] }> {
    const model = 'gemini-2.5-flash';
    const geminiHistory = buildGeminiHistory(history);

    try {
        const aiClient = getAIClient();
        const chat: Chat = aiClient.chats.create({
            model: model,
            history: geminiHistory,
            config: {
                systemInstruction: SYSTEM_PROMPT,
                tools: [{ googleSearch: {} }],
            },
        });

        const stream = await chat.sendMessageStream({ message: newMessage });

        for await (const chunk of stream) {
            const groundingChunks = chunk.candidates?.[0]?.groundingMetadata?.groundingChunks;
            const sources = groundingChunks?.map((chunk: any) => ({
                uri: chunk.web.uri,
                title: chunk.web.title,
            })).filter((source: any) => source.uri && source.title) ?? [];

            yield {
                text: chunk.text,
                sources: sources,
            };
        }
    } catch (error) {
        console.error("Error getting chat response:", error);
        yield {
            text: "I'm sorry, I encountered an error. Please try again.",
            sources: [],
        };
    }
};

export async function* getDeepDiveResponseStream(newMessage: string, file?: File): AsyncGenerator<{ text: string; sources: GroundingSource[] }> {
    const model = 'gemini-2.5-pro';

    const userParts: Part[] = [];

    if (file) {
        const filePart = await fileToGenerativePart(file);
        userParts.push(filePart);
    }
    userParts.push({ text: newMessage || "Please provide a summary of the attached document." });


    const req: GenerateContentParameters = {
        model: model,
        contents: [{ role: 'user', parts: userParts }],
        config: {
            systemInstruction: SYSTEM_PROMPT,
            thinkingConfig: { thinkingBudget: 32768 },
        },
    };

    try {
        const aiClient = getAIClient();
        const stream = await aiClient.models.generateContentStream(req);
        for await (const chunk of stream) {
            yield {
                text: chunk.text,
                sources: [],
            };
        }
    } catch (error) {
        console.error("Error getting deep dive response:", error);
        yield {
            text: "I'm sorry, I encountered an error during the deep dive analysis. Please try again.",
            sources: [],
        };
    }
};