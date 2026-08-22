"use client";

import { useEffect, useRef, useState } from "react";
import type { CityKey } from "@/lib/types";

export default function CitySelector({
  selected,
  onChange,
  cityLabels,
  order,
  availability,
}: {
  selected: CityKey;
  onChange: (city: CityKey) => void;
  cityLabels: Record<CityKey, string>;
  order: CityKey[];
  availability: Record<CityKey, boolean>;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="relative z-30" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="glass-card flex items-center gap-2 px-4 py-2.5 text-sm font-semibold text-[#f5f2ea] hover:border-[rgba(56,189,248,0.35)] transition-colors"
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span aria-hidden="true">📍</span>
        <span>{cityLabels[selected]}</span>
        <span
          className="text-xs text-[#9fb0c8] transition-transform duration-200"
          style={{ transform: open ? "rotate(180deg)" : "rotate(0deg)" }}
          aria-hidden="true"
        >
          ▾
        </span>
      </button>

      {open && (
        <div
          role="listbox"
          className="absolute left-0 z-50 mt-2 w-48 overflow-hidden rounded-2xl py-1"
          style={{
            backgroundColor: "#0d1b30",
            border: "1px solid rgba(255,255,255,0.14)",
            boxShadow: "0 20px 48px rgba(0,0,0,0.6)",
          }}
        >
          {order.map((key) => (
            <button
              key={key}
              role="option"
              aria-selected={key === selected}
              onClick={() => {
                onChange(key);
                setOpen(false);
              }}
              className={`flex w-full items-center justify-between px-4 py-2.5 text-left text-sm transition-colors hover:bg-white/10 ${
                key === selected ? "font-semibold text-[#7dd3c0]" : "text-[#e5e2d8]"
              }`}
            >
              <span>{cityLabels[key]}</span>
              {!availability[key] && (
                <span className="text-[10px] uppercase tracking-wide text-[#7a8ba3]">pending</span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
