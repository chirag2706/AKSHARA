import { useEffect, useState } from "react";
import { connect } from "react-redux";
import Popup from "reactjs-popup";
import { getNotifications } from "../scripts/user";
import "../styles/navnotification.css";

let Navnotification = (props) => {
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    async function load() {
      let notificationsList = await getNotifications(props.auth.token);
      setNotifications(notificationsList);
    }
    load();
  }, []);

  return (
    <Popup
      trigger={
        <button className="notbutton">
          <i className="fas fa-bell bellstyle mt-2 pt-1"></i>
        </button>
      }
      position="bottom center"
    >
      <div>
        {notifications.map((value, i) => (
          <div
            className="card border-dark mb-1"
            style={{ maxWidth: "18rem" }}
            key={i}
          >
            <div className="card-body text-dark">
              <h5 className="card-title">{value.title}</h5>
              <p className="card-text">Displaying requests</p>
            </div>
          </div>
        ))}
      </div>
    </Popup>
  );
};

let mapStateToProps = (state) => ({
  auth: state.auth,
});

export default connect(mapStateToProps)(Navnotification);
