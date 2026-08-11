import { tool } from "@opencode-ai/plugin"

const MAX_RESULTS = 20
const MAX_FETCH_TEXT = 14000
const UA =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

function htmlDecode(s: string): string {
  return s
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#0?39;/g, "'")
    .replace(/&nbsp;/g, " ")
}

function stripTags(s: string): string {
  return s.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim()
}

async function fetchText(url: string, init: RequestInit = {}): Promise<string> {
  const res = await fetch(url, {
    redirect: "follow",
    headers: { "User-Agent": UA, "Accept-Language": "en-US,en;q=0.9" },
    ...init,
  })
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText} for ${url}`)
  return await res.text()
}

// Convert a raw HTML page into readable plain text (best-effort).
function htmlToText(html: string): string {
  const text = html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<noscript[\s\S]*?<\/noscript>/gi, " ")
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(/<li[^>]*>/gi, "\n- ")
    .replace(/<\/(p|div|h[1-6]|ul|ol|tr|section|article|table)>/gi, "\n")
    .replace(/<(br|hr)[^>]*>/gi, "\n")
    .replace(/<[^>]+>/g, " ")
  return htmlDecode(stripTags(text)).replace(/\s*\n\s*/g, "\n").replace(/\n{3,}/g, "\n\n").trim()
}

// Fetch one page and return its readable text (truncated).
async function fetchPageText(url: string): Promise<string> {
  const html = await fetchText(url)
  const text = htmlToText(html)
  return text.length > MAX_FETCH_TEXT ? text.slice(0, MAX_FETCH_TEXT) + "\n...[truncated]" : text
}

type Result = { title: string; url: string; snippet: string }

// DuckDuckGo (HTML endpoint, no API key). Primary.
async function duckDuckGo(query: string): Promise<Result[]> {
  const body = new URLSearchParams({ q: query, b: "", i: "us-en" }).toString()
  const html = await fetchText("https://html.duckduckgo.com/html/", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  })
  const blocks = html.split('<h2 class="result__title">').slice(1)
  const out: Result[] = []
  for (const block of blocks) {
    if (out.length >= MAX_RESULTS) break
    const link = block.match(/class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)<\/a>/s)
    const snip = block.match(/class="result__snippet"[^>]*>(.*?)<\/a>/s)
    if (!link) continue
    out.push({
      title: stripTags(htmlDecode(link[2])),
      url: htmlDecode(link[1]),
      snippet: snip ? stripTags(htmlDecode(snip[1])) : "",
    })
  }
  return out
}

// Bing (fallback)
async function bing(query: string): Promise<Result[]> {
  const html = await fetchText(`https://www.bing.com/search?q=${encodeURIComponent(query)}&setlang=en`)
  const blocks = html.split('<li class="b_algo"').slice(1)
  const out: Result[] = []
  for (const block of blocks) {
    if (out.length >= MAX_RESULTS) break
    const link = block.match(/<h2><a[^>]*href="([^"]+)"[^>]*>(.*?)<\/a><\/h2>/)
    const snip = block.match(/<p[^>]*>(.*?)<\/p>/s)
    if (!link) continue
    out.push({
      title: stripTags(htmlDecode(link[2])),
      url: htmlDecode(link[1]),
      snippet: snip ? stripTags(htmlDecode(snip[1])) : "",
    })
  }
  return out
}

export default tool({
  description:
    "Live web search (DuckDuckGo, then Bing) returning ranked results with title, URL and snippet. ALSO: pass `urls` to fetch the full readable text of specific pages directly (no search needed), or pass `fetch_first` to also pull full text for the top N results. Use for current events and anything past the knowledge cutoff.",
  args: {
    query: tool.schema.string().optional().describe("Search query. Omit if you only want to fetch specific `urls`."),
    num: tool.schema.number().optional().describe("Max search results to list (default 8, up to 20)"),
    urls: tool.schema.array(tool.schema.string()).optional().describe("Fetch full page text for these specific URLs instead of searching"),
    fetch_first: tool.schema.number().optional().describe("Also fetch full text of the first N search results (0-5, default 0)"),
  },
  async execute(args) {
    const errors: string[] = []
    const chunks: string[] = []

    // Mode 1: fetch specific URLs directly (no search).
    if (args.urls && args.urls.length > 0) {
      for (const url of args.urls) {
        try {
          const text = await fetchPageText(url)
          chunks.push(`URL: ${url}\nTITLE: (fetched)\n\n${text}`)
        } catch (e) {
          errors.push(`${url}: ${e instanceof Error ? e.message : String(e)}`)
          chunks.push(`Failed to fetch ${url}: ${e instanceof Error ? e.message : String(e)}`)
        }
      }
      const header = `Fetched ${chunks.filter((c) => !c.startsWith("Failed")).length}/${args.urls.length} requested pages.`
      return `${header}\n\n${chunks.join("\n\n" + "-".repeat(60) + "\n\n")}`
    }

    // Mode 2: web search, optionally deepening a few top results.
    if (!args.query) {
      return "Provide either `query` (to search) or `urls` (to fetch specific pages)."
    }

    const limit = Math.max(1, Math.min(MAX_RESULTS, Math.round(args.num ?? 8)))
    const deepen = Math.max(0, Math.min(5, Math.round(args.fetch_first ?? 0)))
    let results: Result[] = []

    const providers: Array<[string, () => Promise<Result[]>]> = [
      ["DuckDuckGo", () => duckDuckGo(args.query!)],
      ["Bing", () => bing(args.query!)],
    ]
    for (const [name, fn] of providers) {
      try {
        results = await fn()
        if (results.length > 0) break
        errors.push(`${name}: no results`)
      } catch (e) {
        errors.push(`${name}: ${e instanceof Error ? e.message : String(e)}`)
      }
    }

    if (results.length === 0) {
      return (
        `No usable search results for: '${args.query}'\n` +
        `Provider errors: ${errors.join(" | ")}\n` +
        "The search backend may be temporarily blocked. Do NOT conclude data is unavailable just because this failed; you may retry or use a known authoritative source."
      )
    }

    const listed = results.slice(0, limit).map((r, i) => `${i + 1}. ${r.title}\n   ${r.url}\n   ${r.snippet || "(no snippet)"}`)
    chunks.push(`Search results for: '${args.query}'\n${listed.join("\n")}`)

    for (let i = 0; i < Math.min(deepen, results.length); i++) {
      const r = results[i]
      try {
        const text = await fetchPageText(r.url)
        chunks.push(`FULL TEXT of "#${i + 1}: ${r.title}" (${r.url}):\n${text}`)
      } catch (e) {
        chunks.push(`(could not fetch full text of ${r.url}: ${e instanceof Error ? e.message : String(e)})`)
      }
    }

    return chunks.join("\n\n" + "-".repeat(60) + "\n\n")
  },
})