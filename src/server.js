import "dotenv/config";
import express from "express";
import multer from "multer";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { answerQuestion, indexDocument } from "./rag.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const upload = multer({ dest: path.join(__dirname, "..", "uploads") });

app.use(express.json());
app.use(express.static(path.join(__dirname, "..", "public")));

app.post("/api/index", upload.single("document"), async (req, res) => {
  const uploadedPath = req.file?.path;
  try {
    if (!req.file) {
      res.status(400).json({ error: "No file uploaded" });
      return;
    }

    const ext = path.extname(req.file.originalname || "").toLowerCase();
    const mime = req.file.mimetype || "";
    const isPdf = ext === ".pdf" || mime === "application/pdf";
    const isText = ext === ".txt" || mime === "text/plain";
    if (!isPdf && !isText) {
      res.status(400).json({
        error: "Unsupported file type. Use .pdf or .txt",
        details: { ext, mime, name: req.file.originalname },
      });
      return;
    }

    const result = await indexDocument(
      req.file.path,
      req.file.originalname,
      req.file.mimetype
    );
    if (!result.indexed) {
      res.json({ message: "Document already indexed.", alreadyIndexed: true });
      return;
    }

    res.json({
      message: "Indexing completed.",
      chunks: result.chunks,
      fileName: result.fileName,
    });
  } catch (error) {
    console.error("Indexing error:", error);
    const message = error.message || "Indexing failed";
    const status = message.includes("Qdrant is not reachable") ? 503 : 500;
    res.status(status).json({ error: message });
  } finally {
    if (uploadedPath) {
      await fs.unlink(uploadedPath).catch(() => undefined);
    }
  }
});

app.post("/api/ask", async (req, res) => {
  try {
    const question = req.body?.question?.trim();
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
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
