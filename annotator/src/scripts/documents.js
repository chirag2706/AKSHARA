import axios from "axios";
import { server } from "../constants/constant";

async function getDocuments(token, project_id) {
    let response = await axios.post(server + "document-list", {'project': project_id}, {
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

async function uploadDocument(token, data) {
    let response = await axios.post(server + "documents", data, {
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

export { getDocuments, uploadDocument }