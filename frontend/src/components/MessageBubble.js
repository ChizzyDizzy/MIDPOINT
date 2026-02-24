import React, { useState } from "react";

const MessageBubble = ({ message }) => {
  const [isSpeaking, setIsSpeaking] = useState(false);

  const formatTime = (date) => {
    return new Date(date).toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const handleListen = () => {
    if (isSpeaking) {
      // Stop speaking
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }

    const utterance = new SpeechSynthesisUtterance(message.text);
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    // Try to use a natural-sounding English voice
    const voices = window.speechSynthesis.getVoices();
    const preferred = voices.find(
      (v) => v.lang.startsWith("en") && v.name.toLowerCase().includes("female")
    ) || voices.find(
      (v) => v.lang.startsWith("en")
    );
    if (preferred) {
      utterance.voice = preferred;
    }

    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    setIsSpeaking(true);
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className={`message ${message.type}`}>
      <div className='message-bubble'>{message.text}</div>
      <div className='message-meta'>
        <span className='message-time'>{formatTime(message.timestamp)}</span>
        {message.type === "bot" && (
          <button
            className='listen-button'
            onClick={handleListen}
            title={isSpeaking ? "Stop" : "Listen"}
          >
            {isSpeaking ? "■ Stop" : "🔊 Listen"}
          </button>
        )}
      </div>
    </div>
  );
};

export default MessageBubble;
