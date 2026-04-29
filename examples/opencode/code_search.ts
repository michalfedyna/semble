import { tool } from "@opencode-ai/plugin"

const modeSchema = tool.schema.enum(["hybrid", "semantic", "bm25"])

async function runSemble(args: string[], cwd: string) {
  const bin = process.env.SEMBLE_BIN ?? "semble"
  const proc = Bun.spawn([bin, ...args], { cwd, stdout: "pipe", stderr: "pipe" })
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ])
  if (exitCode !== 0) return `Semble failed (${exitCode}):\n${stderr || stdout}`
  return stdout.trim()
}

export default tool({
  description: "Search the current codebase with Semble code search. Prefer this over grep for implementation discovery.",
  args: {
    query: tool.schema.string().describe("Natural-language or code search query"),
    path: tool.schema.string().optional().describe("Codebase path or git URL. Defaults to the current worktree."),
    mode: modeSchema.optional().describe("Search mode: hybrid, semantic, or bm25"),
    top_k: tool.schema.number().int().positive().optional().describe("Number of snippets to return"),
  },
  async execute(args, context) {
    const sourcePath = args.path ?? context.worktree ?? context.directory
    return runSemble([
      "search",
      args.query,
      "--path",
      sourcePath,
      "--mode",
      args.mode ?? "hybrid",
      "--top-k",
      String(args.top_k ?? 5),
    ], context.directory)
  },
})
