import { useState } from "react";
import { connect } from "react-redux";
import { useHistory } from "react-router-dom/cjs/react-router-dom.min";
import {
    authAction,
    LOGIN_SUCCESS,
    LOGIN_FAIL,
    REGISTER_SUCCESS,
    REGISTER_FAIL,
} from "../actions/authAction";
import { doLogin, doRegister } from "../scripts/authenticate";
import '../styles/login.css'

const Login = (props) => {
    let [message, set_message] = useState("");
    let { auth, dispatch } = props;
    let history = useHistory();

    let [userDetails, setUserDetails] = useState({
        name: "",
        username: "",
        password: "",
        rememberme: false,
    });

    let handleName = (e) => {
        setUserDetails({ ...userDetails, name: e.target.value });
    };

    let handleUserName = (e) => {
        setUserDetails({ ...userDetails, username: e.target.value });
    };

    let handlePassword = (e) => {
        setUserDetails({ ...userDetails, password: e.target.value });
    };

    let handleRememberMe = (e) => {
        const { checked } = e.target;
        console.log(checked);
        setUserDetails({ ...userDetails, rememberme: checked });
    };

    let handleLogin = async (e) => {
        e.preventDefault();
        let res = await doLogin(
            { username: userDetails.username, password: userDetails.password },
            userDetails.rememberme
        );
        res.payload.username = userDetails.username
        dispatch(authAction(res.type, res.payload));
        if (res.type === LOGIN_SUCCESS) {
            history.push("/");
        } else if (res.type === LOGIN_FAIL) {
            set_message("Incorrect username or password");
        }
    };

    let handleRegister = async (e) => {
        e.preventDefault();
        let res = await doRegister({
            name: userDetails.name,
            username: userDetails.username,
            password: userDetails.password,
        });
        res.payload.username = userDetails.username
        dispatch(authAction(res.type, res.payload));
        if (res.type === REGISTER_SUCCESS) {
            history.push("/");
        } else if (res.type === REGISTER_FAIL) {
            set_message("username already taken");
        }
    };

    let Formlogin = (
        <form onSubmit={(e) => handleLogin(e)}>
            <div className="mb-3">
                <label htmlFor="login-username" className="form-label">
                    Username
                </label>
                <input
                    type="text"
                    className="form-control"
                    id="login-username"
                    onChange={(e) => handleUserName(e)}
                    value={userDetails.username}
                ></input>
            </div>
            <div className="mb-3">
                <label htmlFor="login-password" className="form-label">
                    Password
                </label>
                <input
                    type="password"
                    className="form-control"
                    id="login-password"
                    onChange={(e) => handlePassword(e)}
                    value={userDetails.password}
                ></input>
            </div>
            <div className="mb-3 form-check">
                <input
                    type="checkbox"
                    className="form-check-input"
                    id="login-remember"
                    onChange={(e) => handleRememberMe(e)}
                    defaultChecked={userDetails.rememberme}
                ></input>
                <label className="form-check-label" htmlFor="login-remember">
                    Remember me
                </label>
            </div>
            <button type="submit" className="btn btn-primary">
                Submit
            </button>
        </form>
    );

    const Formsignup = (
        <form onSubmit={(e) => handleRegister(e)}>
            <div className="mb-3">
                <label htmlFor="regester-name" className="form-label">
                    Name
                </label>
                <input
                    type="text"
                    className="form-control"
                    id="regester-name"
                    onChange={(e) => handleName(e)}
                    value={userDetails.name}
                ></input>
            </div>
            <div className="mb-3">
                <label htmlFor="register-username" className="form-label">
                    Username
                </label>
                <input
                    type="text"
                    className="form-control"
                    id="register-username"
                    onChange={(e) => handleUserName(e)}
                    value={userDetails.username}
                ></input>
            </div>
            <div className="mb-3">
                <label htmlFor="resgister-pass" className="form-label">
                    Password
                </label>
                <input
                    type="password"
                    className="form-control"
                    id="resgister-pass"
                    defaultChecked={userDetails.password}
                    onChange={(e) => handlePassword(e)}
                    value={userDetails.password}
                ></input>
            </div>
            <button type="submit" className="btn btn-primary">
                Submit
            </button>
        </form>
    );

    return (
        <div>
        <div className="auth-centered">
            <ul className="nav nav-pills mb-3" id="pills-tab" role="tablist">
                <li className="nav-item" role="presentation">
                    <button
                        className="nav-link active"
                        id="pills-home-tab"
                        data-bs-toggle="pill"
                        data-bs-target="#pills-home"
                        type="button"
                        role="tab"
                        aria-controls="pills-home"
                        aria-selected="true"
                    >
                        Login
                    </button>
                </li>
                <li className="nav-item" role="presentation">
                    <button
                        className="nav-link"
                        id="pills-profile-tab"
                        data-bs-toggle="pill"
                        data-bs-target="#pills-profile"
                        type="button"
                        role="tab"
                        aria-controls="pills-profile"
                        aria-selected="false"
                    >
                        SignUp
                    </button>
                </li>
            </ul>
            <div className="tab-content" id="pills-tabContent">
                <div
                    className="tab-pane fade show active"
                    id="pills-home"
                    role="tabpanel"
                    aria-labelledby="pills-home-tab"
                >
                    {Formlogin}
                </div>
                <div
                    className="tab-pane fade"
                    id="pills-profile"
                    role="tabpanel"
                    aria-labelledby="pills-profile-tab"
                >
                    {Formsignup}
                </div>
            </div>
        </div>
        </div>
    );
};

const mapStateToProps = (state) => ({
    auth: state.auth,
});

export default connect(mapStateToProps)(Login);
