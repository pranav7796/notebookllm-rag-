import "dotenv/config";
import fs from "node:fs/promises";
import path from "node:path";
import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { answerQuestion, getConfig, indexDocument } from "./rag.js";

async function ensureFileExists(filePath) {
  const normalized = path.resolve(filePath);
  await fs.access(normalized);
  return normalized;
}

async function runIndex(filePath) {
  const normalizedPath = await ensureFileExists(filePath);
  await indexDocument(normalizedPath);
  console.log("Indexing completed.");
}

async function runAsk(question) {
  const result = await answerQuestion(question);
  console.log("\nAnswer:\n" + result.answer);
}

async function runInteractive() {
  const rl = createInterface({ input, output });

  try {
    const config = getConfig();
    console.log("NotebookLM RAG (CLI)");
    console.log("Qdrant:", config.qdrantUrl);
    console.log("Collection:", config.collectionName);

    const shouldIndex = (await rl.question("Index a document? (y/n): ")).trim().toLowerCase();
    if (shouldIndex === "y") {
      const filePath = await rl.question("Enter path to .pdf or .txt: ");
      await runIndex(filePath.trim());
    }

    while (true) {
      const question = await rl.question("\nAsk a question (or press Enter to exit): ");
      if (!question.trim()) {
        break;
      }
      await runAsk(question.trim());
    }
  } finally {
    rl.close();
  }
}

async function main() {
  const [command, ...rest] = process.argv.slice(2);

  if (command === "index") {
    if (!rest[0]) {
      throw new Error("Usage: node src/index.js index <path-to-pdf-or-txt>");
    }
    await runIndex(rest[0]);
    return;
  }

  if (command === "ask") {
    if (!rest.length) {
      throw new Error("Usage: node src/index.js ask <question>");
    }
    await runAsk(rest.join(" "));
    return;
  }

  await runInteractive();
}

main().catch((error) => {
  console.error("\nError:", error.message);
  process.exit(1);
});
