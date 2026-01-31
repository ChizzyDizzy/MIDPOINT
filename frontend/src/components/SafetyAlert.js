import React from "react";
import { AlertTriangle, Phone, Heart } from "lucide-react";

const SafetyAlert = ({ safetyData, onClose }) => {
  return (
    <>
      <div className='safety-alert-overlay' onClick={onClose} />{" "}
      <div className='safety-alert'>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            marginBottom: "20px",
          }}
        >
          <AlertTriangle
            size={24}
            color='#e74c3c'
            style={{ marginRight: "10px" }}
          />{" "}
          <h2> Important Safety Information </h2>{" "}
        </div>
        <p>
          {" "}
          Your wellbeing is our top priority.Based on what you 've shared, we
          want to ensure you have access to immediate support.
        </p>
        <div className='emergency-numbers'>
          <h3> Crisis Support(Sri Lanka) </h3>{" "}
          <p>
            {" "}
            <Phone size={16} /> National Crisis Line: 1333
          </p>
          <p>
            {" "}
            <Phone size={16} /> Mental Health Helpline: 1926
          </p>
          <p>
            {" "}
            <Phone size={16} /> Emergency Services: 119
          </p>
        </div>
        <div style={{ marginTop: "20px" }}>
          <p>
            {" "}
            <Heart
              size={16}
              style={{ display: "inline", marginRight: "5px" }}
            />
            Remember: You are not alone.These feelings are temporary, and help
            is available.{" "}
          </p>{" "}
        </div>
        <button
          onClick={onClose}
          style={{
            marginTop: "20px",
            padding: "12px 24px",
            background: "linear-gradient(180deg, #9b7ed8, #7c3aed)",
            color: "#fff",
            border: "2px solid #4c1d95",
            boxShadow: "inset 1px 1px 0px #c4b5fd, 2px 2px 0px #1e1b4b",
            fontFamily: "'Press Start 2P', cursive",
            fontSize: "0.55rem",
            cursor: "pointer",
            textTransform: "uppercase",
            letterSpacing: "1px",
          }}
        >
          I understand{" "}
        </button>{" "}
      </div>{" "}
    </>
  );
};

export default SafetyAlert;
