import fs from "node:fs/promises";
import path from "node:path";
import { indexDocument } from "../src/rag.js";
import { parseMultipartForm } from "./_utils.js";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  let uploadedPath;

  try {
    const { files } = await parseMultipartForm(req);
    const upload = Array.isArray(files.document) ? files.document[0] : files.document;

    if (!upload) {
      res.status(400).json({ error: "No file uploaded" });
      return;
    }

    uploadedPath = upload.filepath;
    const originalName = upload.originalFilename || "";
    const mimeType = upload.mimetype || "";
    const ext = path.extname(originalName).toLowerCase();
    const isPdf = ext === ".pdf" || mimeType === "application/pdf";
    const isText = ext === ".txt" || mimeType === "text/plain";

    if (!isPdf && !isText) {
      res.status(400).json({
        error: "Unsupported file type. Use .pdf or .txt",
        details: { ext, mime: mimeType, name: originalName },
      });
      return;
    }

    const result = await indexDocument(uploadedPath, originalName, mimeType);

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
}