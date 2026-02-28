interface AlgoBrawlLogoProps {
    className?: string
    /** Compact mode for the header — smaller, fewer particles, no tagline */
    compact?: boolean
}

export function AlgoBrawlLogo({ className }: AlgoBrawlLogoProps) {
    // simple text logo
    return (
        <div className={`flex items-center justify-center font-bold ${className || ""}`}>
            AlgoBrawl
        </div>
    )
}
