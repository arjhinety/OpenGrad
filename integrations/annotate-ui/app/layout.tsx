import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OpenGrad annotate",
  description: "Local human annotation for OpenGrad research datasets",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
