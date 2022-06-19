import axios from "axios";
import { server } from "../constants/constant";

async function getNotifications(token) {
    let response = await axios.get(server + "get-notifications", {
        headers: {
            authtoken: token,
        },
    });

    if (response.status === 200) {
        let notifications = response.data["notifications"];
        return notifications;
    } else return false;
}

async function getUserDetails(token) {
    let response = await axios.get(server + "get-user", {
        headers: {
            authtoken: token,
        },
    });

    if (response.status === 200) {
        return response.data;
    } else return false;
}

async function updateUserDetails(token, userDetails) {
    let response = await axios.put(server + "get-user", userDetails, {
        headers: {
            authtoken: token
        },
    }).catch(function() {
        return false;
    });

    if (response.status === 200) {
        return true;
    } else return false;
}

async function getRequests(token, data) {
    let response = await axios.post(server + "get-requests", data, {
        headers: {
            authtoken: token
        }
    }).catch(function () {
        return null
    });
    if(response.status === 200)
        return response.data['requests'].reverse()
    else
        return null
}

async function getProjectsList(token) {
    let response = await axios.get(server + "get-project-list", {
        headers: {
            authtoken: token
        }
    }).catch(function () {
        return null
    });
    if(response.status === 200)
        return response.data['project-list']
    else
        return null
}

async function sendRequest(token, newRequest) {
    let response = await axios.post(server + "request-list", newRequest, {
        headers: {
            authtoken: token
        },
    }).catch(function() {
        return false;
    });

    if (response.status === 200) {
        return true;
    } else return false;
}

async function sendRequestAction(token, id, req_status) {
    let response = await axios.put(server + "request-list", { 'id': id.toString(), 'status': req_status }, {
        headers: {
            authtoken: token
        },
    }).catch(function() {
        return false;
    });

    if (response.status === 200) {
        return true;
    } else return false;
}

export { getNotifications, getUserDetails, updateUserDetails, getRequests, getProjectsList, sendRequest, sendRequestAction };
