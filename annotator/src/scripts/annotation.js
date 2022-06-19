import axios from 'axios'
import { server } from '../constants/constant'

async function fetchAllAnnotationsOfAParticularProject(projectDetails,authToken) {
    let response = await axios.post(server + 'all-annotations',projectDetails, {
        headers: {
            authtoken: authToken
        },
    })
    .catch(function(error) {
        return {
            status:404,
            data:{
                "annotations": []
            }
        }
    })
    return response.data;

}

export { fetchAllAnnotationsOfAParticularProject };