import React from "react";

const MessageBubble = ({ message }) => {
  const formatTime = (date) => {
    return new Date(date).toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className={`message ${message.type}`}>
      <div className='message-bubble'> {message.text} </div>{" "}
      <div className='message-time'> {formatTime(message.timestamp)} </div>{" "}
    </div>
  );
};

export default MessageBubble;
