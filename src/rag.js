import "dotenv/config";
import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantClient } from "@qdrant/js-client-rest";
import { QdrantVectorStore } from "@langchain/qdrant";
import { OpenAI } from "openai";

const DEFAULT_QDRANT_URL = "http://localhost:6333";
const DEFAULT_COLLECTION = "notebookllm";
const DEFAULT_QDRANT_API_KEY = "";
const DEFAULT_OPENAI_MODEL = "gpt-4.1-mini";
const DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large";
const DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1";
const DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free";
const DEFAULT_OPENROUTER_EMBEDDING_MODEL = "text-embedding-3-large";
const DEFAULT_RETRIEVAL_K = 4;
const DEFAULT_SCORE_THRESHOLD = 0.2;

function getEnv(key, fallback) {
  return process.env[key] || fallback;
}

export function getConfig() {
  return {
    qdrantUrl: getEnv("QDRANT_URL", DEFAULT_QDRANT_URL),
    collectionName: getEnv("QDRANT_COLLECTION", DEFAULT_COLLECTION),
    qdrantApiKey: getEnv("QDRANT_API_KEY", DEFAULT_QDRANT_API_KEY),
    provider: getEnv("LLM_PROVIDER", ""),
    openaiApiKey: getEnv("OPENAI_API_KEY", ""),
    openaiBaseUrl: getEnv("OPENAI_BASE_URL", ""),
    openaiModel: getEnv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
    openaiEmbeddingModel: getEnv(
      "OPENAI_EMBEDDING_MODEL",
      DEFAULT_OPENAI_EMBEDDING_MODEL
    ),
    openrouterApiKey: getEnv("OPENROUTER_API_KEY", ""),
    openrouterBaseUrl: getEnv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_URL),
    openrouterModel: getEnv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
    openrouterEmbeddingModel: getEnv(
      "OPENROUTER_EMBEDDING_MODEL",
      DEFAULT_OPENROUTER_EMBEDDING_MODEL
    ),
    retrievalK: Number(getEnv("RETRIEVAL_K", DEFAULT_RETRIEVAL_K)),
    scoreThreshold: Number(getEnv("SCORE_THRESHOLD", DEFAULT_SCORE_THRESHOLD)),
  };
}

function buildQdrantClient(config) {
  return new QdrantClient({
    url: config.qdrantUrl,
    apiKey: config.qdrantApiKey || undefined,
  });
}

async function ensureFileHashIndex(client, collectionName) {
  try {
    await client.createPayloadIndex(collectionName, {
      field_name: "fileHash",
      field_schema: "keyword",
    });
  } catch (error) {
    const message = String(error?.message || "").toLowerCase();
    if (message.includes("already exists") || message.includes("conflict")) {
      return;
    }
    throw error;
  }
}

function resolveProvider(config) {
  if (config.provider) {
    return config.provider.toLowerCase();
  }

  if (config.openaiApiKey) {
    return "openai";
  }

  return "openrouter";
}

function getProviderConfig(config, mode) {
  const provider = resolveProvider(config);

  if (provider === "openai") {
    if (!config.openaiApiKey) {
      throw new Error("Missing OPENAI_API_KEY in environment");
    }

    return {
      provider,
      apiKey: config.openaiApiKey,
      baseURL: config.openaiBaseUrl || undefined,
      model: mode === "embeddings" ? config.openaiEmbeddingModel : config.openaiModel,
    };
  }

  if (!config.openrouterApiKey) {
    throw new Error("Missing OPENROUTER_API_KEY in environment");
  }

  return {
    provider: "openrouter",
    apiKey: config.openrouterApiKey,
    baseURL: config.openrouterBaseUrl,
    model:
      mode === "embeddings"
        ? config.openrouterEmbeddingModel
        : config.openrouterModel,
  };
}

async function hashFile(filePath) {
  const buffer = await fs.readFile(filePath);
  return crypto.createHash("sha256").update(buffer).digest("hex");
}

function buildEmbeddings(config) {
  const providerConfig = getProviderConfig(config, "embeddings");

  return new OpenAIEmbeddings({
    model: providerConfig.model,
    apiKey: providerConfig.apiKey,
    configuration: providerConfig.baseURL
      ? { baseURL: providerConfig.baseURL }
      : undefined,
  });
}

function buildChatClient(config) {
  const providerConfig = getProviderConfig(config, "chat");

  return {
    client: new OpenAI({
      apiKey: providerConfig.apiKey,
      baseURL: providerConfig.baseURL,
    }),
    model: providerConfig.model,
  };
}

export async function loadDocuments(filePath, originalName, mimeType) {
  const extSource = originalName || filePath;
  const ext = path.extname(extSource).toLowerCase();

  console.log("loadDocuments:", {
    extSource,
    ext,
    mimeType,
  });

  if (ext === ".pdf" || mimeType === "application/pdf") {
    const loader = new PDFLoader(filePath);
    return loader.load();
  }

  if (ext === ".txt" || mimeType === "text/plain") {
    const loader = new TextLoader(filePath);
    return loader.load();
  }

  try {
    const handle = await fs.open(filePath, "r");
    const buffer = Buffer.alloc(4);
    await handle.read(buffer, 0, 4, 0);
    await handle.close();
    const signature = buffer.toString("utf8");
    console.log("loadDocuments signature:", signature);
    if (signature === "%PDF") {
      const loader = new PDFLoader(filePath);
      return loader.load();
    }
  } catch (error) {
    throw error;
  }

  throw new Error("Unsupported file type. Use .pdf or .txt");
}

export async function chunkDocuments(documents, metadata) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const chunks = await splitter.splitDocuments(documents);

  return chunks.map((chunk, index) => ({
    ...chunk,
    metadata: {
      ...chunk.metadata,
      ...metadata,
      chunkIndex: index + 1,
      totalChunks: chunks.length,
    },
  }));
}

async function isAlreadyIndexed(config, fileHash) {
  const client = buildQdrantClient(config);
  const { collectionName } = config;
  try {
    const collections = await client.getCollections();
    const exists = collections.collections?.some(
      (collection) => collection.name === collectionName
    );

    if (!exists) {
      return false;
    }

    await ensureFileHashIndex(client, collectionName);

    const result = await client.scroll(collectionName, {
      limit: 1,
      filter: {
        must: [{ key: "fileHash", match: { value: fileHash } }],
      },
      with_payload: true,
    });

    return (result?.points?.length || 0) > 0;
  } catch (error) {
    if (String(error?.message || "").includes("doesn't exist")) {
      return false;
    }
    throw error;
  }
}

async function collectionExists(config) {
  const client = buildQdrantClient(config);
  const { collectionName } = config;
  try {
    const collections = await client.getCollections();
    return collections.collections?.some(
      (collection) => collection.name === collectionName
    );
  } catch (error) {
    throw new Error("Qdrant is not reachable. Check QDRANT_URL and server status.");
  }
}

export async function indexDocument(filePath, originalName, mimeType) {
  const config = getConfig();
  const fileHash = await hashFile(filePath);
  const fileName = originalName || path.basename(filePath);

  const alreadyIndexed = await isAlreadyIndexed(config, fileHash);
  if (alreadyIndexed) {
    return { indexed: false, reason: "already_indexed", fileName };
  }

  const documents = await loadDocuments(filePath, originalName, mimeType);
  const chunks = await chunkDocuments(documents, {
    source: filePath,
    fileName,
    fileHash,
  });

  const embeddings = buildEmbeddings(config);
  const client = buildQdrantClient(config);

  await QdrantVectorStore.fromDocuments(chunks, embeddings, {
    client,
    collectionName: config.collectionName,
  });

  return { indexed: true, chunks: chunks.length, fileName };
}

export async function answerQuestion(question) {
  const config = getConfig();

  if (!question.trim()) {
    throw new Error("Question cannot be empty");
  }

  const exists = await collectionExists(config);
  if (!exists) {
    return {
      answer: "I could not find this in the document.",
      sources: [],
    };
  }

  const embeddings = buildEmbeddings(config);
  const client = buildQdrantClient(config);
  const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
    client,
    collectionName: config.collectionName,
  });

  const scored = await vectorStore.similaritySearchWithScore(
    question,
    config.retrievalK
  );

  const filtered = scored.filter(([, score]) =>
    typeof score === "number" ? score >= config.scoreThreshold : true
  );

  if (!filtered.length) {
    return {
      answer: "I could not find this in the document.",
      sources: [],
    };
  }

  const context = filtered
    .map(([chunk], index) => {
      const page = chunk.metadata?.loc?.pageNumber ?? chunk.metadata?.pageNumber;
      const pageInfo = page ? `page ${page}` : "unknown page";
      const fileName = chunk.metadata?.fileName || "unknown file";
      return `Chunk ${index + 1} (${fileName}, ${pageInfo}):\n${chunk.pageContent}`;
    })
    .join("\n\n");

  const { client: chatClient, model } = buildChatClient(config);

  const systemPrompt = [
    "You are a strict assistant for question answering over a document.",
    "Use ONLY the provided context from the document.",
    "If the answer is not in the context, reply: 'I could not find this in the document.'",
    "Cite chunk numbers in your answer like [Chunk 1].",
    "Context:",
    context,
  ].join("\n");

  const response = await chatClient.chat.completions.create({
    model,
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: question },
    ],
  });

  return {
    answer: response.choices?.[0]?.message?.content?.trim() || "",
    sources: filtered.map(([chunk, score], index) => ({
      chunk: index + 1,
      score,
      page: chunk.metadata?.loc?.pageNumber ?? chunk.metadata?.pageNumber ?? null,
      fileName: chunk.metadata?.fileName || null,
      content: chunk.pageContent,
    })),
  };
}
