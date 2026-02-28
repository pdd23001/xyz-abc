import React from "react"

interface BenchwarmerLogoProps {
    className?: string
    /** Compact mode for the header — smaller, fewer particles, no tagline */
    compact?: boolean
}

export function BenchwarmerLogo({ className, compact = false }: BenchwarmerLogoProps) {
    // simple text logo
    return (
        <div className={`flex items-center justify-center font-bold ${className || ""}`}>
            AI Algo Metric
        </div>
    )
}
