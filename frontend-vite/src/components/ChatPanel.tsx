import { MessageList } from "@/components/chat/MessageList"
import { ChatInput } from "@/components/chat/ChatInput"
import { Message } from "@/hooks/use-chat"
import { ScrollArea } from "@/components/ui/scroll-area"

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
    if (isWelcome) {
        return (
            <div className="flex flex-col h-full relative group">
                {/* Subtle background glow */}
                <div className="absolute inset-0 bg-gradient-to-tr from-sky-300/10 via-transparent to-blue-300/10 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none" />

                <div className="flex-1" />

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
