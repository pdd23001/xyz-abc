import { BenchwarmerLogo } from "@/components/BenchwarmerLogo"
import { cn } from "@/lib/utils"

export function HeroHeader({ className }: { className?: string }) {
    return (
        <div className={cn("relative flex flex-col items-center justify-center pt-10 pb-6 lg:pt-16 lg:pb-10", className)}>
            {/* Animated Glow Background behind Header */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none overflow-hidden">
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[300px] bg-sky-200/40 rounded-[100%] blur-[100px] animate-pulse" />
                <div className="absolute top-[40%] left-[45%] -translate-x-1/2 -translate-y-1/2 w-[500px] h-[400px] bg-blue-100/50 rounded-[100%] blur-[100px] animate-pulse" style={{ animationDelay: '1.5s' }} />
            </div>

            <div className="relative z-10 flex flex-col items-center gap-6">
                <BenchwarmerLogo className="w-[340px] sm:w-[480px] drop-shadow-2xl" />
                <div className="text-center space-y-2 mt-4">
                    <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-foreground/90 transform transition-all hover:scale-[1.01]">
                        Welcome to Benchwarmer
                    </h1>
                    <p className="text-muted-foreground/80 max-w-lg mx-auto text-sm sm:text-base font-medium">
                        Upload your research papers and algorithm to begin extracting, benchmarking, and optimizing interactive models side-by-side.
                    </p>
                </div>
            </div>
        </div>
    )
}
