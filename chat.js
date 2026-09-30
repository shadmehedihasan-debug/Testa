// Vercel serverless function: lets the web page answer questions that are NOT in the FAQ.
// The API key stays on the server (set ANTHROPIC_API_KEY in Vercel -> Settings -> Environment Variables).
const SYSTEM = `You are a helpful assistant in an online store's help chat.
The customer asked something our FAQ does not cover. Answer ANY question helpfully, accurately and concisely using general knowledge.
If the message is a greeting or casual chat, reply naturally and friendly, and offer to help.
Rules:
- Never invent store-specific facts (policies, prices, stock, delivery times, account details). If the question depends on them, say you don't have that information and suggest emailing support@example.com.
- If you are not sure about something, say so instead of guessing.
- Keep replies under 150 words unless the customer asks for detail.`;

module.exports = async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "Use POST" });
  const key = process.env.ANTHROPIC_API_KEY;
  if (!key) return res.status(500).json({ error: "ANTHROPIC_API_KEY is not set" });

  let body = req.body;
  if (typeof body === "string") { try { body = JSON.parse(body); } catch { body = {}; } }
  let msgs = ((body && body.messages) || []).slice(-5).map(m => ({
    role: m.role === "assistant" ? "assistant" : "user",
    content: String(m.content || "").slice(0, 500),
  }));
  while (msgs.length && msgs[0].role !== "user") msgs.shift();      // must start with a user turn
  if (!msgs.length || msgs[msgs.length - 1].role !== "user") return res.status(400).json({ error: "Send a user message" });

  try {
    const r = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01" },
      body: JSON.stringify({ model: "claude-haiku-4-5-20251001", max_tokens: 500, temperature: 0, system: SYSTEM, messages: msgs }),
    });
    if (!r.ok) return res.status(502).json({ error: "Upstream error " + r.status });
    const data = await r.json();
    const answer = (data.content || []).filter(b => b.type === "text").map(b => b.text).join("").trim();
    return res.status(200).json({ answer });
  } catch (e) {
    return res.status(502).json({ error: "Request failed" });
  }
};
