const LOGIN_SUCCESS = "LOGIN_SUCCESS";
const LOGIN_FAIL = "LOGIN_FAIL";
const LOGOUT = "LOGOUT";
const REGISTER_SUCCESS = "REGISTER_SUCCESS";
const REGISTER_FAIL = "REGISTER_FAIL";
const SET_ALERT = "SET_ALERT";
const REMOVE_ALERT = "REMOVE_ALERT";

function authAction(type, payload) {
    return {
        type: type,
        payload: payload,
    };
}

export {
    LOGIN_SUCCESS,
    LOGIN_FAIL,
    LOGOUT,
    REGISTER_SUCCESS,
    REGISTER_FAIL,
    SET_ALERT,
    REMOVE_ALERT,
    authAction,
};
