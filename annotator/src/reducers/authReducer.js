import {
    LOGIN_SUCCESS,
    LOGIN_FAIL,
    LOGOUT,
    REGISTER_SUCCESS,
    REGISTER_FAIL,
} from "./../actions/authAction";

const state = JSON.parse( sessionStorage.getItem('state'));
const initialState = state ?? {
    token: null,
    username: null,
    isAuthenticated: false,
    rememberme: false,
    message: null
};

export default function authReducer(state = initialState, action) {
    const { type, payload } = action;

    switch (type) {
        case LOGIN_SUCCESS:
            var temp = {
                ...state,
                token: payload.token,
                username: payload.username,
                isAuthenticated: true
            };
            if (payload.rememberme)
                localStorage.setItem("state", JSON.stringify(payload));
            sessionStorage.setItem("state", JSON.stringify(payload));
            return temp;
        case LOGIN_FAIL:
            localStorage.removeItem("state");
            sessionStorage.removeItem("state");
            return {
                ...state,
                token: null,
                isAuthenticated: false,
                message: payload.message
            };
        case REGISTER_SUCCESS:
            var temp = {
                ...state,
                token: payload.token,
                username: payload.username,
                isAuthenticated: true
            };
            sessionStorage.setItem("state", JSON.stringify(payload));
            return temp;
        case REGISTER_FAIL:
            return {
                ...state,
                token: null,
                isAuthenticated: false,
            };
        case LOGOUT:
            localStorage.removeItem("state");
            sessionStorage.removeItem("state");
            return {
                ...state,
                token: null,
                isAuthenticated: false,
            };
        default:
            return state;
    }
}