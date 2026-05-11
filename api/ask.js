import { answerQuestion } from "../src/rag.js";
import { readJsonBody } from "./_utils.js";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  try {
    const body = await readJsonBody(req);
    const question = body?.question?.trim();

    if (!question) {
      res.status(400).json({ error: "Question is required" });
      return;
    }

    const result = await answerQuestion(question);
    res.json({ answer: result.answer, sources: result.sources || [] });
  } catch (error) {
    console.error("Ask error:", error);
    const message = error.message || "Request failed";
    const status = message.includes("Qdrant is not reachable") ? 503 : 500;
    res.status(status).json({ error: message });
  }
}