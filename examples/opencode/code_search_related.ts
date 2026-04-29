import { tool } from "@opencode-ai/plugin"

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
  description: "Find code chunks related to a file and line using Semble code search.",
  args: {
    file_path: tool.schema.string().describe("File path from a prior code_search result"),
    line: tool.schema.number().int().positive().describe("Line number in the indexed file"),
    path: tool.schema.string().optional().describe("Codebase path or git URL. Defaults to the current worktree."),
    top_k: tool.schema.number().int().positive().optional().describe("Number of snippets to return"),
  },
  async execute(args, context) {
    const sourcePath = args.path ?? context.worktree ?? context.directory
    return runSemble([
      "related",
      "--file",
      args.file_path,
      "--line",
      String(args.line),
      "--path",
      sourcePath,
      "--top-k",
      String(args.top_k ?? 5),
    ], context.directory)
  },
})
