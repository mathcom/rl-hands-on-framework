import React, { useState } from 'react';

interface CodeBlockProps {
  code: string;
  filename: string;
  language?: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ code, filename, language = "python" }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy!", err);
    }
  };

  return (
    <div className="rounded-lg overflow-hidden border border-slate-700 bg-slate-900 shadow-xl my-4 flex flex-col h-full">
      <div className="flex items-center justify-between px-4 py-2 bg-slate-800 border-b border-slate-700">
        <span className="font-mono text-sm text-slate-300 font-semibold">{filename}</span>
        <button
          onClick={handleCopy}
          className={`text-xs px-3 py-1 rounded transition-colors duration-200 font-medium ${
            copied
              ? "bg-green-600 text-white"
              : "bg-slate-700 text-slate-300 hover:bg-slate-600"
          }`}
        >
          {copied ? "Copied!" : "Copy Code"}
        </button>
      </div>
      <div className="overflow-auto p-4 flex-grow bg-slate-950/50">
        <pre className="font-mono text-sm leading-relaxed text-slate-200 whitespace-pre">
          {code}
        </pre>
      </div>
    </div>
  );
};

export default CodeBlock;