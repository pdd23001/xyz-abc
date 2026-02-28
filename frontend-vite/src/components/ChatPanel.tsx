import { MessageList } from "@/components/chat/MessageList"
import { ChatInput } from "@/components/chat/ChatInput"
import { Message } from "@/hooks/use-chat"
import { Sparkles } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"

const EXAMPLE_PROMPTS = [
    "Extract the greedy algorithm from the PDF.",
    "Benchmark my LP relaxation against the paper.",
    "Find edge cases where my algorithm fails.",
    "What is the time complexity of the SDP approach?",
]

interface ChatPanelProps {
    messages: Message[]
    isLoading: boolean
    onSend: (text: string, files: { pdfs?: File[]; algorithm?: File }) => void
    onAlgorithmSelect?: (indices: number[]) => void
    onChoiceSelect?: (choiceId: string, values: string[]) => void
    isWelcome?: boolean

    // Pass-through props for ChatInput
    pdfs?: File[]
    onPdfsChange?: (pdfs: File[]) => void
    algo?: File
    onAlgoChange?: (algo: File | undefined) => void
    llmBackend?: string
    setLlmBackend?: (val: string) => void
    executionMode?: string
    setExecutionMode?: (val: string) => void
    bottomLeftContent?: React.ReactNode
}

export function ChatPanel({
    messages,
    isLoading,
    onSend,
    onAlgorithmSelect,
    onChoiceSelect,
    isWelcome,
    pdfs,
    onPdfsChange,
    algo,
    onAlgoChange,
    bottomLeftContent
}: ChatPanelProps) {
    const handleExampleClick = (prompt: string) => {
        onSend(prompt, { pdfs, algorithm: algo })
    }

    if (isWelcome) {
        return (
            <div className="flex flex-col h-full relative group">
                {/* Subtle background glow */}
                <div className="absolute inset-0 bg-gradient-to-tr from-sky-300/10 via-transparent to-blue-300/10 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none" />

                <div className="flex-1 flex flex-col justify-end pb-8 px-8 space-y-4">
                    <div className="space-y-2 translate-y-4 opacity-0 animate-in fade-in slide-in-from-bottom-6 duration-700 fill-auto" style={{ animationDelay: "200ms", animationFillMode: "forwards" }}>
                        <div className="flex items-center gap-2 text-slate-500 mb-3 px-1.5 object-left">
                            <Sparkles className="w-4 h-4 text-sky-500" />
                            <span className="text-sm font-medium">Example Prompts</span>
                        </div>
                        <div className="flex flex-wrap gap-2">
                            {EXAMPLE_PROMPTS.map((prompt, i) => (
                                <button
                                    key={i}
                                    onClick={() => handleExampleClick(prompt)}
                                    className="text-left text-[13px] px-3.5 py-2 rounded-xl bg-white/60 backdrop-blur-sm border border-white/80 text-slate-600 hover:text-sky-800 hover:bg-white/90 hover:border-sky-300/50 transition-all duration-300 hover:-translate-y-0.5 hover:shadow-[0_4px_20px_rgba(14,165,233,0.1)]"
                                >
                                    {prompt}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                <div className="h-[200px] flex-shrink-0 animate-in fade-in slide-in-from-bottom-8 duration-700 z-10" style={{ animationDelay: "100ms" }}>
                    <ChatInput
                        onSend={onSend}
                        disabled={isLoading}
                        placeholder="Describe your problem or what algorithm to run..."
                        hideAttachButtons
                        className="h-full rounded-2xl border-white/60 shadow-[0_8px_32px_rgba(0,0,0,0.02)] backdrop-blur-xl bg-white/50"
                        pdfs={pdfs}
                        onPdfsChange={onPdfsChange}
                        algo={algo}
                        onAlgoChange={onAlgoChange}
                        bottomLeftContent={bottomLeftContent}
                    />
                </div>
            </div>
        )
    }

    return (
        <div className="flex-1 min-w-0 flex flex-col h-full bg-background/50">
            <div className="flex-1 min-h-0 relative">
                <ScrollArea className="h-full w-full">
                    <MessageList
                        messages={messages}
                        isLoading={isLoading}
                        onAlgorithmSelect={onAlgorithmSelect}
                        onChoiceSelect={onChoiceSelect}
                    />
                </ScrollArea>
                {/* Gradient overlay at bottom of scroll area for smooth fade */}
                <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-background to-transparent pointer-events-none" />
            </div>

            <div className="flex-shrink-0 p-4 max-w-3xl w-full mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500 bg-background/80 backdrop-blur-sm z-10">
                <ChatInput
                    onSend={onSend}
                    disabled={isLoading}
                    className="shadow-[0_8px_32px_rgba(0,0,0,0.02)] backdrop-blur-xl bg-white/60 border-white/80 focus-within:ring-2 focus-within:ring-sky-500/20 focus-within:border-sky-400/50 transition-all duration-300 rounded-xl"
                />
            </div>
        </div>
    )
}
