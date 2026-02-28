import { useRef, useState } from "react"
import { X } from "lucide-react"
import { cn } from "@/lib/utils"
import { cva, type VariantProps } from "class-variance-authority"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

const uploadCardVariants = cva(
    "relative flex flex-col items-center justify-center gap-3 rounded-2xl border px-4 py-8 cursor-pointer transition-all duration-300 ease-out group overflow-hidden bg-white/60 backdrop-blur-md shadow-[0_8px_32px_rgba(0,0,0,0.02)]",
    {
        variants: {
            variant: {
                default:
                    "border-white/80 hover:border-sky-300/60 hover:bg-white/80 hover:shadow-[0_4px_20px_rgba(14,165,233,0.1)] hover:-translate-y-1",
                active:
                    "border-sky-400 bg-sky-500/10 scale-[1.02] shadow-[0_4px_25px_rgba(14,165,233,0.2)]",
                success:
                    "border-emerald-500/50 bg-emerald-500/10 hover:border-emerald-400 hover:shadow-[0_4px_20px_rgba(16,185,129,0.15)] hover:-translate-y-1",
            },
        },
        defaultVariants: {
            variant: "default",
        },
    }
)

interface UploadCardProps extends React.HTMLAttributes<HTMLDivElement>, VariantProps<typeof uploadCardVariants> {
    accept: string
    multiple?: boolean
    icon: React.ReactNode
    title: string
    subtitle: string
    helperText?: string
    files: File[]
    onFilesChange: (files: File[]) => void
}

export function UploadCard({
    accept,
    multiple = false,
    icon,
    title,
    subtitle,
    helperText,
    files,
    onFilesChange,
    className,
    ...props
}: UploadCardProps) {
    const inputRef = useRef<HTMLInputElement>(null)
    const [isDragOver, setIsDragOver] = useState(false)

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragOver(true)
    }

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragOver(false)
    }

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragOver(false)

        const droppedFiles = Array.from(e.dataTransfer.files)
        const extensions = accept.split(",").map((a) => a.trim().toLowerCase())
        const filtered = droppedFiles.filter((f) =>
            extensions.some((ext) => f.name.toLowerCase().endsWith(ext))
        )
        if (filtered.length === 0) return

        if (multiple) {
            onFilesChange([...files, ...filtered])
        } else {
            onFilesChange([filtered[0]])
        }
    }

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files) return
        const selected = Array.from(e.target.files)
        if (multiple) {
            onFilesChange([...files, ...selected])
        } else {
            onFilesChange(selected.slice(0, 1))
        }
        e.target.value = ""
    }

    const removeFile = (idx: number) => {
        onFilesChange(files.filter((_, i) => i !== idx))
    }

    const variant = isDragOver ? "active" : files.length > 0 ? "success" : "default"

    return (
        <TooltipProvider>
            <Tooltip delayDuration={300}>
                <TooltipTrigger asChild>
                    <div
                        onClick={() => inputRef.current?.click()}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        className={cn(uploadCardVariants({ variant }), className)}
                        {...props}
                    >
                        {/* Subtle background glow effect */}
                        <div className="absolute inset-0 bg-gradient-to-br from-sky-400/10 via-transparent to-blue-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />

                        <input
                            ref={inputRef}
                            type="file"
                            accept={accept}
                            multiple={multiple}
                            className="hidden"
                            onChange={handleInputChange}
                        />

                        <div className={cn(
                            "transition-transform duration-300",
                            isDragOver ? "scale-110 text-sky-500" : "text-slate-500/60 group-hover:text-sky-500 group-hover:scale-105"
                        )}>
                            {icon}
                        </div>
                        <div className="text-center relative z-10 w-full px-2">
                            <div className="text-[15px] font-semibold text-foreground/90 tracking-tight">{title}</div>
                            <div className="text-[12px] text-muted-foreground/60 mt-1 font-medium">{subtitle}</div>
                            {helperText && (
                                <div className="text-[10px] text-muted-foreground/40 mt-1.5">{helperText}</div>
                            )}
                        </div>

                        {/* Uploaded file chips */}
                        {files.length > 0 && (
                            <div className="flex flex-wrap gap-2 mt-2 w-full justify-center relative z-10" onClick={(e) => e.stopPropagation()}>
                                {files.map((f, i) => (
                                    <div
                                        key={i}
                                        className="flex items-center gap-1.5 bg-background/80 border border-border/50 shadow-sm px-2.5 py-1.5 rounded-md text-[11px] font-medium max-w-full animate-in zoom-in-95 duration-200"
                                    >
                                        <span className="truncate max-w-[120px] text-foreground/80">{f.name}</span>
                                        <button
                                            type="button"
                                            onClick={(e) => {
                                                e.stopPropagation()
                                                removeFile(i)
                                            }}
                                            className="text-muted-foreground hover:text-red-400 hover:bg-red-400/10 p-0.5 rounded-sm transition-colors flex-shrink-0"
                                        >
                                            <X size={12} />
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </TooltipTrigger>
                <TooltipContent side="bottom" sideOffset={12} className="px-3 py-1.5 text-xs text-muted-foreground bg-secondary/95 backdrop-blur-md border-border">
                    <p>Click or drag and drop to attach files</p>
                </TooltipContent>
            </Tooltip>
        </TooltipProvider>
    )
}
