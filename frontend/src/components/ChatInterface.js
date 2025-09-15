import React, { useState, useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";
import SafetyAlert from "./SafetyAlert";
import { sendMessage } from "../services/api";

function ChatInterface(props) {
  const onMoodUpdate = props.onMoodUpdate;
  const onEmergency = props.onEmergency;

  const [messages, setMessages] = useState([
    {
      type: "bot",
      text: "Hello! I am SafeMind, your mental health support companion. How are you feeling today?",
      timestamp: new Date(),
    },
  ]);
  const [inputText, setInputText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [showSafetyAlert, setShowSafetyAlert] = useState(false);
  const [safetyData, setSafetyData] = useState(null);
  const messagesEndRef = useRef(null);

  function scrollToBottom() {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }

  useEffect(
    function () {
      scrollToBottom();
    },
    [messages]
  );

  async function handleSubmit(e) {
    e.preventDefault();
    if (!inputText.trim()) return;

    const userMessage = {
      type: "user",
      text: inputText,
      timestamp: new Date(),
    };

    setMessages(function (prev) {
      return [...prev, userMessage];
    });
    setInputText("");
    setIsTyping(true);

    try {
      const response = await sendMessage(inputText, sessionId);

      if (!sessionId) {
        setSessionId(response.session_id);
      }

      if (
        response.safety.risk_level === "immediate" ||
        response.safety.risk_level === "high"
      ) {
        setSafetyData(response.safety);
        setShowSafetyAlert(true);
        if (onEmergency) {
          onEmergency();
        }
      }

      setTimeout(function () {
        const botMessage = {
          type: "bot",
          text: response.response,
          timestamp: new Date(),
          safety: response.safety,
        };
        setMessages(function (prev) {
          return [...prev, botMessage];
        });
        setIsTyping(false);

        if (onMoodUpdate) {
          onMoodUpdate({
            timestamp: new Date(),
            sentiment:
              response.safety.risk_level === "none" ? "neutral" : "concerned",
          });
        }
      }, 1000);
    } catch (error) {
      console.error("Error sending message:", error);
      setIsTyping(false);
      const errorMessage = {
        type: "bot",
        text: "I apologize, but I am having trouble connecting right now. Please try again.",
        timestamp: new Date(),
      };
      setMessages(function (prev) {
        return [...prev, errorMessage];
      });
    }
  }

  return React.createElement(
    "div",
    { className: "chat-interface" },
    React.createElement(
      "div",
      { className: "chat-header" },
      React.createElement("h2", null, "Chat with SafeMind"),
      React.createElement("p", null, "Your conversation is private and secure")
    ),
    React.createElement(
      "div",
      { className: "chat-messages" },
      messages.map(function (message, index) {
        return React.createElement(MessageBubble, {
          key: index,
          message: message,
        });
      }),
      isTyping &&
        React.createElement(
          "div",
          { className: "message bot" },
          React.createElement(
            "div",
            { className: "typing-indicator" },
            React.createElement("span", { className: "typing-dot" }),
            React.createElement("span", { className: "typing-dot" }),
            React.createElement("span", { className: "typing-dot" })
          )
        ),
      React.createElement("div", { ref: messagesEndRef })
    ),
    React.createElement(
      "div",
      { className: "chat-input-container" },
      React.createElement(
        "form",
        { onSubmit: handleSubmit, className: "chat-input-form" },
        React.createElement("input", {
          type: "text",
          value: inputText,
          onChange: function (e) {
            setInputText(e.target.value);
          },
          placeholder: "Type your message here...",
          className: "chat-input",
          disabled: isTyping,
        }),
        React.createElement(
          "button",
          { type: "submit", className: "send-button", disabled: isTyping },
          "Send"
        )
      )
    ),
    showSafetyAlert &&
      React.createElement(SafetyAlert, {
        safetyData: safetyData,
        onClose: function () {
          setShowSafetyAlert(false);
        },
      })
  );
}

export default ChatInterface;
