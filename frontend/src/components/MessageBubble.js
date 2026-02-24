import React, { useState, useEffect } from "react";
import { Volume2, Square, Bot } from "lucide-react";

const MessageBubble = ({ message }) => {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceGender, setVoiceGender] = useState("female");
  const [voices, setVoices] = useState([]);

  useEffect(() => {
    const loadVoices = () => {
      setVoices(window.speechSynthesis.getVoices());
    };
    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;
    return () => {
      window.speechSynthesis.onvoiceschanged = null;
    };
  }, []);

  const formatTime = (date) => {
    return new Date(date).toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getVoice = (gender) => {
    const enVoices = voices.filter((v) => v.lang.startsWith("en"));

    if (gender === "male") {
      return (
        enVoices.find((v) => /\b(male|daniel|james|david|mark|guy)\b/i.test(v.name)) ||
        enVoices.find((v) => !/\b(female|samantha|karen|victoria|fiona|zira|hazel)\b/i.test(v.name)) ||
        enVoices[0]
      );
    }

    return (
      enVoices.find((v) => /\b(female|samantha|karen|victoria|fiona|zira|hazel)\b/i.test(v.name)) ||
      enVoices[0]
    );
  };

  const handleListen = () => {
    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }

    const utterance = new SpeechSynthesisUtterance(message.text);
    utterance.rate = 0.95;
    utterance.pitch = voiceGender === "female" ? 1.1 : 0.85;
    utterance.volume = 1.0;

    const selectedVoice = getVoice(voiceGender);
    if (selectedVoice) {
      utterance.voice = selectedVoice;
    }

    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    setIsSpeaking(true);
    window.speechSynthesis.speak(utterance);
  };

  const isBot = message.type === "bot";

  return (
    <div className={`message ${message.type}`}>
      <div className="message-row">
        {isBot && (
          <div className="bot-avatar">
            <Bot size={20} />
          </div>
        )}
        <div className="message-content">
          <div className="message-bubble">{message.text}</div>
          <div className="message-meta">
            <span className="message-time">{formatTime(message.timestamp)}</span>
            {isBot && (
              <div className="listen-controls">
                <select
                  className="voice-select"
                  value={voiceGender}
                  onChange={(e) => setVoiceGender(e.target.value)}
                  title="Choose voice"
                >
                  <option value="female">Female</option>
                  <option value="male">Male</option>
                </select>
                <button
                  className="listen-button"
                  onClick={handleListen}
                  title={isSpeaking ? "Stop" : "Listen"}
                >
                  {isSpeaking ? (
                    <>
                      <Square size={12} /> Stop
                    </>
                  ) : (
                    <>
                      <Volume2 size={14} /> Listen
                    </>
                  )}
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;
