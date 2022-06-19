import axios from "axios";
import { server } from "../constants/constant";

async function getUsersList(token) {
    let response = await axios.get(server + "get-user-list", {
        headers: {
            authtoken: token
        }
    }).catch(function () {
        return null
    });
    if(response.status === 200)
        return response.data['user-list']
    else
        return null
}

async function getUserProjects(token) {
    let response = await axios.get(server + "get-user-projects", {
        headers: {
            authtoken: token
        }
    }).catch(function () {
        return null
    });
    if(response.status === 200)
        return response.data
    else
        return null
}

async function createProject(token, data) {
    let response = await axios.post(server + "create-project", data, {
        headers: {
            authtoken: token
        }
    }).catch(function () {
        return null
    });
    if(response.status === 200)
        return response.data
    else
        return null
}

async function getProjectUsers(token, id) {
    let response = await axios.post(server + "get-project-users", {'project_id': id}, {
        headers: {
            authtoken: token
        }
    }).catch(function () {
        return null
    });
    if(response.status === 200)
        return response.data
    else
        return null
}

async function deleteUser(token, username, project_id) {
    let response = await axios.post(server + "delete-user", {
        'user': username,
        'project_id': project_id
    }, {
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

async function leaveProject(token, project_id) {
    let response = await axios.post(server + "leave-project", {
        'project_id': project_id
    }, {
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

export { getUsersList, getUserProjects, createProject,  getProjectUsers, deleteUser, leaveProject }
