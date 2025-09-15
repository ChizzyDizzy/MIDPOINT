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
            padding: "10px 20px",
            background: "#667eea",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer",
          }}
        >
          I understand{" "}
        </button>{" "}
      </div>{" "}
    </>
  );
};

export default SafetyAlert;
